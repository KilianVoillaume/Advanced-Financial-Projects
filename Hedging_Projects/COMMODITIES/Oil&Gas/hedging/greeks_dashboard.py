"""
hedging/greeks_dashboard.py - Real-Time Greeks Dashboard

This module provides comprehensive Greeks visualization and monitoring
for options portfolios in commodity hedging.
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


class GreeksDashboard:
    """
    Real-time Greeks dashboard for options positions.
    
    Provides comprehensive visualization and monitoring of portfolio Greeks
    with professional-grade charts and risk metrics.
    """
    
    @staticmethod
    def _calculate_position_greeks(position) -> Dict[str, float]:
        """ Calculate Greeks for a single position """
        try:
            if position.strategy != "Options" or not position.strike_price:
                return {
                    'delta': 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            
            current_price = position.current_price
            strike = position.strike_price
            time_to_exp = time_to_expiration(3)  # 3 months
            risk_free_rate = get_risk_free_rate()
            volatility = get_commodity_volatility(position.commodity)
            
            # Determine option type based on position
            if position.size > 0:  # Long underlying -> need put protection
                option_type = 'put'
            else:  # Short underlying -> need call protection
                option_type = 'call'
            
            greeks = BlackScholesCalculator.calculate_greeks(
                current_price, strike, time_to_exp, risk_free_rate, volatility, option_type
            )
            
            position_multiplier = abs(position.size) * position.hedge_ratio
            
            return {
                'delta': greeks['delta'] * position_multiplier,
                'gamma': greeks['gamma'] * position_multiplier,
                'theta': greeks['theta'] * position_multiplier,
                'vega': greeks['vega'] * position_multiplier,
                'rho': greeks['rho'] * position_multiplier
            }
            
        except Exception as e:
            st.error(f"Error calculating Greeks for position: {e}")
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    @staticmethod
    def create_greeks_heatmap(portfolio_positions: Dict) -> go.Figure:
        """ Create heatmap of Greeks across all positions """
        
        if not portfolio_positions:
            # Empty heatmap
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Portfolio Greeks Heatmap",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=400
            )
            return fig
        
        greeks_data = []
        position_names = []
        
        for name, position in portfolio_positions.items():
            if position.strategy == "Options":
                greeks = GreeksDashboard._calculate_position_greeks(position)
                greeks_data.append([
                    greeks['delta'],
                    greeks['gamma'],
                    greeks['theta'],
                    greeks['vega'],
                    greeks['rho']
                ])
                position_names.append(name)
        
        if not greeks_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Portfolio Greeks Heatmap",
                height=400
            )
            return fig
        
        greeks_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
        greeks_array = np.array(greeks_data)
        
        normalized_data = np.zeros_like(greeks_array)
        for i in range(greeks_array.shape[1]):
            col = greeks_array[:, i]
            if np.std(col) > 0:
                normalized_data[:, i] = (col - np.mean(col)) / np.std(col)
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized_data,
            x=greeks_names,
            y=position_names,
            colorscale='RdBu',
            zmid=0,
            text=greeks_array,
            texttemplate='%{text:.3f}',
            textfont={"size": 12, "color": "white"},
            hoverongaps=False,
            colorbar=dict(title="Normalized Greeks")
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
            font=dict(family="Inter", size=12)
        )
        
        return fig
    
    @staticmethod
    def create_delta_exposure_chart(portfolio_positions: Dict) -> go.Figure:
        """ Create delta exposure visualization """
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Delta Exposure", height=350)
            return fig
        
        delta_by_commodity = {}
        position_deltas = []
        position_names = []
        
        for name, position in portfolio_positions.items():
            if position.strategy == "Options":
                greeks = GreeksDashboard._calculate_position_greeks(position)
                delta = greeks['delta']
                
                commodity = position.commodity
                if commodity not in delta_by_commodity:
                    delta_by_commodity[commodity] = 0
                delta_by_commodity[commodity] += delta
                
                position_deltas.append(delta)
                position_names.append(name)
        
        if not position_deltas:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Delta Exposure", height=350)
            return fig
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Delta by Position", "Delta by Commodity"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Chart 1: Delta by position
        colors = ['#FF6B6B' if delta < 0 else '#4ECDC4' for delta in position_deltas]
        
        fig.add_trace(
            go.Bar(
                x=position_names,
                y=position_deltas,
                name="Position Delta",
                marker_color=colors,
                text=[f'{delta:.3f}' for delta in position_deltas],
                textposition='auto',
                showlegend=False
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
                'text': "Delta Exposure Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            height=350,
            showlegend=False,
            font=dict(family="Inter", size=10)
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
        
        return fig
    
    @staticmethod
    def create_gamma_risk_chart(portfolio_positions: Dict) -> go.Figure:
        """ Create gamma risk (convexity) visualization """
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Gamma Risk", height=350)
            return fig
        
        gamma_data = []
        position_names = []
        
        for name, position in portfolio_positions.items():
            if position.strategy == "Options":
                greeks = GreeksDashboard._calculate_position_greeks(position)
                gamma = greeks['gamma']
                
                gamma_data.append({
                    'position': name,
                    'gamma': gamma,
                    'commodity': position.commodity,
                    'size': abs(position.size),
                    'hedge_ratio': position.hedge_ratio
                })
        
        if not gamma_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Gamma Risk", height=350)
            return fig
        
        # Create gamma risk chart
        df = pd.DataFrame(gamma_data)
        
        fig = go.Figure()
        
        commodities = df['commodity'].unique()
        colors = ['#667eea', '#764ba2', '#4ECDC4', '#FF6B6B', '#48bb78']
        
        for i, commodity in enumerate(commodities):
            commodity_data = df[df['commodity'] == commodity]
            
            fig.add_trace(go.Scatter(
                x=commodity_data['position'],
                y=commodity_data['gamma'],
                mode='markers',
                name=commodity,
                marker=dict(
                    size=commodity_data['size'] / 50,  # Scale bubble size
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                text=[f"Gamma: {gamma:.4f}<br>Size: {size:,.0f}<br>Hedge: {hedge:.1%}" 
                      for gamma, size, hedge in zip(commodity_data['gamma'], 
                                                   commodity_data['size'], 
                                                   commodity_data['hedge_ratio'])],
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             '%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': "Gamma Risk Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            xaxis_title="Position",
            yaxis_title="Gamma",
            height=350,
            showlegend=True,
            font=dict(family="Inter", size=12)
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    @staticmethod
    def create_theta_decay_chart(portfolio_positions: Dict) -> go.Figure:
        """ Create time decay (theta) visualization """
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Theta Decay", height=350)
            return fig
        
        theta_data = []
        position_names = []
        
        for name, position in portfolio_positions.items():
            if position.strategy == "Options":
                greeks = GreeksDashboard._calculate_position_greeks(position)
                theta = greeks['theta']
                
                theta_data.append(theta)
                position_names.append(name)
        
        if not theta_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Theta Decay", height=350)
            return fig
        
        fig = go.Figure()
        
        # Cumulative theta decay over 30 days
        days = np.arange(0, 31)
        cumulative_theta = np.zeros(len(days))
        
        for i, day in enumerate(days):
            cumulative_theta[i] = np.sum(theta_data) * day
        
        fig.add_trace(go.Scatter(
            x=days,
            y=cumulative_theta,
            mode='lines+markers',
            name='Cumulative Theta Decay',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=6, color='#FF6B6B'),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        daily_theta = [np.sum(theta_data)] * len(days)
        fig.add_trace(go.Scatter(
            x=days,
            y=daily_theta,
            mode='lines',
            name='Daily Theta',
            line=dict(color='#667eea', width=2, dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title={
                'text': "Theta Decay Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            xaxis_title="Days",
            yaxis_title="Cumulative P&L Impact ($)",
            height=350,
            font=dict(family="Inter", size=12),
            yaxis2=dict(
                title="Daily Theta ($)",
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
        
        return fig
    
    @staticmethod
    def create_vega_sensitivity_chart(portfolio_positions: Dict) -> go.Figure:
        """ Create volatility sensitivity (vega) visualization """
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Vega Sensitivity", height=350)
            return fig
        
        vega_data = []
        position_names = []
        
        for name, position in portfolio_positions.items():
            if position.strategy == "Options":
                greeks = GreeksDashboard._calculate_position_greeks(position)
                vega = greeks['vega']
                
                vega_data.append({
                    'position': name,
                    'vega': vega,
                    'commodity': position.commodity
                })
        
        if not vega_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No options positions found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Vega Sensitivity", height=350)
            return fig
        
        # Vega sensitivity chart
        df = pd.DataFrame(vega_data)
        
        # Colatility shock scenarios
        vol_shocks = np.arange(-10, 11, 1)  # -10% to +10% vol shocks
        
        fig = go.Figure()
        
        # Each position's vega sensitivity
        colors = ['#4ECDC4', '#FF6B6B', '#48bb78', '#667eea', '#f39c12']
        
        for i, (_, row) in enumerate(df.iterrows()):
            pnl_impact = row['vega'] * vol_shocks
            
            fig.add_trace(go.Scatter(
                x=vol_shocks,
                y=pnl_impact,
                mode='lines+markers',
                name=row['position'],
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=5)
            ))
        
        # Net portfolio vega
        total_vega = df['vega'].sum()
        net_pnl_impact = total_vega * vol_shocks
        
        fig.add_trace(go.Scatter(
            x=vol_shocks,
            y=net_pnl_impact,
            mode='lines+markers',
            name='Net Portfolio',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig.update_layout(
            title={
                'text': "Vega Sensitivity Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            xaxis_title="Volatility Shock (%)",
            yaxis_title="P&L Impact ($)",
            height=350,
            font=dict(family="Inter", size=12),
            hovermode='x unified'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    @staticmethod
    def render_greeks_summary_cards(portfolio_positions: Dict):
        """ Render summary cards for portfolio Greeks """
        
        net_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        options_count = 0
        
        for position in portfolio_positions.values():
            if position.strategy == "Options":
                options_count += 1
                greeks = GreeksDashboard._calculate_position_greeks(position)
                
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value
        
        if options_count == 0:
            st.info("📊 No options positions found. Add options positions to see Greeks analysis.")
            return
        
        # Summary cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta_color = "#4ECDC4" if net_greeks['delta'] >= 0 else "#FF6B6B"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: {delta_color};">Portfolio Delta</div>
                <div class="metric-value">{net_greeks['delta']:.3f}</div>
                <div class="metric-subtitle">Price Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            gamma_color = "#48bb78" if net_greeks['gamma'] >= 0 else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: {gamma_color};">Portfolio Gamma</div>
                <div class="metric-value">{net_greeks['gamma']:.4f}</div>
                <div class="metric-subtitle">Convexity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            theta_color = "#FF6B6B" if net_greeks['theta'] < 0 else "#48bb78"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: {theta_color};">Portfolio Theta</div>
                <div class="metric-value">${net_greeks['theta']:.2f}</div>
                <div class="metric-subtitle">Daily Decay</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            vega_color = "#667eea"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: {vega_color};">Portfolio Vega</div>
                <div class="metric-value">${net_greeks['vega']:.2f}</div>
                <div class="metric-subtitle">Vol Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            rho_color = "#8e44ad"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: {rho_color};">Portfolio Rho</div>
                <div class="metric-value">${net_greeks['rho']:.2f}</div>
                <div class="metric-subtitle">Rate Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Risk Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Delta Risk:**")
            if abs(net_greeks['delta']) < 0.1:
                st.success("✅ Well-hedged portfolio (low delta)")
            elif abs(net_greeks['delta']) < 0.5:
                st.warning("⚠️ Moderate directional risk")
            else:
                st.error("❌ High directional risk")
            
            st.markdown("**⚡ Gamma Risk:**")
            if abs(net_greeks['gamma']) < 0.01:
                st.success("✅ Low convexity risk")
            elif abs(net_greeks['gamma']) < 0.05:
                st.warning("⚠️ Moderate convexity risk")
            else:
                st.error("❌ High convexity risk")
        
        with col2:
            st.markdown("**⏰ Theta Risk:**")
            if net_greeks['theta'] > -10:
                st.success("✅ Low time decay")
            elif net_greeks['theta'] > -50:
                st.warning("⚠️ Moderate time decay")
            else:
                st.error("❌ High time decay")
            
            st.markdown("**🌊 Vega Risk:**")
            if abs(net_greeks['vega']) < 50:
                st.success("✅ Low volatility risk")
            elif abs(net_greeks['vega']) < 200:
                st.warning("⚠️ Moderate volatility risk")
            else:
                st.error("❌ High volatility risk")
    
    @staticmethod
    def render_real_time_greeks_monitor(portfolio_positions: Dict):
        """ Render real-time Greeks monitoring interface """
        
        st.markdown("**⚡ Real-Time Greeks Monitor**")
        
        if st.button("🔄 Refresh Greeks", type="secondary"):
            st.rerun()
        
        greeks_table_data = []
        
        for name, position in portfolio_positions.items():
            if position.strategy == "Options":
                greeks = GreeksDashboard._calculate_position_greeks(position)
                
                greeks_table_data.append({
                    'Position': name,
                    'Commodity': position.commodity,
                    'Size': f"{position.size:,.0f}",
                    'Hedge Ratio': f"{position.hedge_ratio:.1%}",
                    'Delta': f"{greeks['delta']:.3f}",
                    'Gamma': f"{greeks['gamma']:.4f}",
                    'Theta': f"${greeks['theta']:.2f}",
                    'Vega': f"${greeks['vega']:.2f}",
                    'Rho': f"${greeks['rho']:.2f}"
                })
        
        if greeks_table_data:
            df = pd.DataFrame(greeks_table_data)
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                height=300
            )
            
            # Download Greeks report
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Greeks Report",
                data=csv,
                file_name="greeks_report.csv",
                mime="text/csv"
            )
        else:
            st.info("No options positions found for Greeks monitoring.")
    
    @staticmethod
    def create_scenario_analysis_chart(portfolio_positions: Dict) -> go.Figure:
        """ Create scenario analysis for portfolio Greeks """
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Scenario Analysis", height=400)
            return fig
        
        scenarios = [
            {"name": "Base Case", "price_shock": 0.0, "vol_shock": 0.0},
            {"name": "Bull Market", "price_shock": 0.10, "vol_shock": -0.02},
            {"name": "Bear Market", "price_shock": -0.10, "vol_shock": 0.02},
            {"name": "High Vol", "price_shock": 0.0, "vol_shock": 0.05},
            {"name": "Low Vol", "price_shock": 0.0, "vol_shock": -0.05},
            {"name": "Crisis", "price_shock": -0.20, "vol_shock": 0.10}
        ]
        
        # Scenario impacts
        scenario_results = []
        
        for scenario in scenarios:
            total_pnl = 0.0
            
            for position in portfolio_positions.values():
                if position.strategy == "Options":
                    greeks = GreeksDashboard._calculate_position_greeks(position)
                    
                    # P&L impact
                    current_price = position.current_price
                    price_change = current_price * scenario["price_shock"]
                    
                    delta_pnl = greeks['delta'] * price_change
                    gamma_pnl = 0.5 * greeks['gamma'] * (price_change ** 2)
                    vega_pnl = greeks['vega'] * scenario["vol_shock"] * 100
                    
                    position_pnl = delta_pnl + gamma_pnl + vega_pnl
                    total_pnl += position_pnl
            
            scenario_results.append({
                'scenario': scenario['name'],
                'pnl': total_pnl
            })
        
        # Scenario chart
        df = pd.DataFrame(scenario_results)
        
        fig = go.Figure()
        
        colors = ['#4ECDC4' if pnl >= 0 else '#FF6B6B' for pnl in df['pnl']]
        
        fig.add_trace(go.Bar(
            x=df['scenario'],
            y=df['pnl'],
            marker_color=colors,
            text=[f'${pnl:,.0f}' for pnl in df['pnl']],
            textposition='auto',
            name='Scenario P&L'
        ))
        
        fig.update_layout(
            title={
                'text': "Greeks Scenario Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            xaxis_title="Scenario",
            yaxis_title="P&L Impact ($)",
            height=400,
            font=dict(family="Inter", size=12),
            showlegend=False
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig


class GreeksMonitor:
    """ Real-time Greeks monitoring and alerting system """
    
    @staticmethod
    def check_risk_limits(portfolio_positions: Dict) -> Dict[str, List[str]]:
        """ nCheck portfolio Greeks against predefined risk limits """
        alerts = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
        net_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for position in portfolio_positions.values():
            if position.strategy == "Options":
                greeks = GreeksDashboard._calculate_position_greeks(position)
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value
        
        limits = {
            'delta': {'critical': 1.0, 'warning': 0.5},
            'gamma': {'critical': 0.1, 'warning': 0.05},
            'theta': {'critical': -100, 'warning': -50},
            'vega': {'critical': 500, 'warning': 200}
        }
        
        for greek_name, greek_value in net_greeks.items():
            if greek_name in limits:
                limit = limits[greek_name]
                
                if greek_name == 'theta':  # Negative values for theta
                    if greek_value < limit['critical']:
                        alerts['critical'].append(f"High theta decay: ${greek_value:.2f}/day")
                    elif greek_value < limit['warning']:
                        alerts['warning'].append(f"Moderate theta decay: ${greek_value:.2f}/day")
                else:  # Positive values for other Greeks
                    if abs(greek_value) > limit['critical']:
                        alerts['critical'].append(f"High {greek_name} exposure: {greek_value:.3f}")
                    elif abs(greek_value) > limit['warning']:
                        alerts['warning'].append(f"Moderate {greek_name} exposure: {greek_value:.3f}")
        
        # Alerts
        if len(alerts['critical']) == 0 and len(alerts['warning']) == 0:
            alerts['info'].append("All Greeks within acceptable risk limits")
        
        return alerts
    
    @staticmethod
    def render_risk_alerts(portfolio_positions: Dict):
        """ Render risk alerts dashboard """
        
        alerts = GreeksMonitor.check_risk_limits(portfolio_positions)
        
        st.markdown("### 🚨 Risk Alerts")
        
        # Critical alerts
        if alerts['critical']:
            for alert in alerts['critical']:
                st.error(f"🔴 **CRITICAL**: {alert}")
        
        # Warning alerts
        if alerts['warning']:
            for alert in alerts['warning']:
                st.warning(f"🟡 **WARNING**: {alert}")
        
        # Info alerts
        if alerts['info']:
            for alert in alerts['info']:
                st.success(f"🟢 **INFO**: {alert}")
        
        # Risk score calculation
        risk_score = len(alerts['critical']) * 3 + len(alerts['warning']) * 1
        max_risk_score = 12  # Arbitrary max for scaling
        
        st.markdown("### 📊 Overall Risk Score")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Portfolio Risk Level"},
            delta = {'reference': 3},
            gauge = {
                'axis': {'range': [None, max_risk_score]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgreen"},
                    {'range': [3, 6], 'color': "yellow"},
                    {'range': [6, max_risk_score], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6
                }
            }
        ))
        
        fig.update_layout(height=300, font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_greeks_evolution_chart(portfolio_positions: Dict, days_ahead: int = 30) -> go.Figure:
        """ Create chart showing how Greeks evolve over time """
        
        if not portfolio_positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Greeks Evolution", height=400)
            return fig
        
        # Greeks evolution
        days = np.arange(0, days_ahead + 1)
        
        delta_evolution = []
        gamma_evolution = []
        theta_evolution = []
        vega_evolution = []
        
        for day in days:
            daily_delta = 0.0
            daily_gamma = 0.0
            daily_theta = 0.0
            daily_vega = 0.0
            
            for position in portfolio_positions.values():
                if position.strategy == "Options":
                    current_price = position.current_price
                    strike = position.strike_price
                    original_time = time_to_expiration(3)  # 3 months
                    remaining_time = max(original_time - (day / 365.0), 0.01)
                    risk_free_rate = get_risk_free_rate()
                    volatility = get_commodity_volatility(position.commodity)
                    
                    option_type = 'put' if position.size > 0 else 'call'
                    
                    greeks = BlackScholesCalculator.calculate_greeks(
                        current_price, strike, remaining_time, risk_free_rate, volatility, option_type
                    )
                    
                    position_multiplier = abs(position.size) * position.hedge_ratio
                    
                    daily_delta += greeks['delta'] * position_multiplier
                    daily_gamma += greeks['gamma'] * position_multiplier
                    daily_theta += greeks['theta'] * position_multiplier
                    daily_vega += greeks['vega'] * position_multiplier
            
            delta_evolution.append(daily_delta)
            gamma_evolution.append(daily_gamma)
            theta_evolution.append(daily_theta)
            vega_evolution.append(daily_vega)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Delta Evolution", "Gamma Evolution", "Theta Evolution", "Vega Evolution"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Delta
        fig.add_trace(
            go.Scatter(x=days, y=delta_evolution, name="Delta", line=dict(color="#4ECDC4", width=2)),
            row=1, col=1
        )
        
        # Gamma
        fig.add_trace(
            go.Scatter(x=days, y=gamma_evolution, name="Gamma", line=dict(color="#48bb78", width=2)),
            row=1, col=2
        )
        
        # Theta
        fig.add_trace(
            go.Scatter(x=days, y=theta_evolution, name="Theta", line=dict(color="#FF6B6B", width=2)),
            row=2, col=1
        )
        
        # Vega
        fig.add_trace(
            go.Scatter(x=days, y=vega_evolution, name="Vega", line=dict(color="#667eea", width=2)),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': "Greeks Evolution Over Time",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter'}
            },
            height=500,
            showlegend=False,
            font=dict(family="Inter", size=10)
        )
        
        fig.update_xaxes(title_text="Days", row=2, col=1)
        fig.update_xaxes(title_text="Days", row=2, col=2)
        
        return fig


def render_enhanced_greeks_tab(portfolio, analysis_ready):
    """ Greeks monitoring tab for the main application """
    
    st.markdown('<div class="section-header">📈 Real-Time Greeks Monitor</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to see Greeks")
        return
    
    if len(portfolio.positions) == 0:
        st.info("📊 Add positions to start Greeks monitoring")
        return
    
    # Check if any options positions exist
    options_positions = {name: pos for name, pos in portfolio.positions.items() 
                        if pos.strategy == "Options"}
    
    if not options_positions:
        st.info("📊 Add options positions to see Greeks analysis")
        return
    
    GreeksMonitor.render_risk_alerts(portfolio.positions)
    
    st.markdown("---")
    
    GreeksDashboard.render_greeks_summary_cards(portfolio.positions)
    
    st.markdown("---")
    
    st.markdown("### 📊 Greeks Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Current Greeks", "⏰ Time Evolution", "🎯 Scenario Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            delta_chart = GreeksDashboard.create_delta_exposure_chart(portfolio.positions)
            st.plotly_chart(delta_chart, use_container_width=True)
            
            gamma_chart = GreeksDashboard.create_gamma_risk_chart(portfolio.positions)
            st.plotly_chart(gamma_chart, use_container_width=True)
        
        with col2:
            theta_chart = GreeksDashboard.create_theta_decay_chart(portfolio.positions)
            st.plotly_chart(theta_chart, use_container_width=True)
            
            vega_chart = GreeksDashboard.create_vega_sensitivity_chart(portfolio.positions)
            st.plotly_chart(vega_chart, use_container_width=True)
        
        st.markdown("### 🔥 Portfolio Greeks Heatmap")
        heatmap = GreeksDashboard.create_greeks_heatmap(portfolio.positions)
        st.plotly_chart(heatmap, use_container_width=True)
    
    with tab2:
        evolution_chart = GreeksMonitor.create_greeks_evolution_chart(portfolio.positions)
        st.plotly_chart(evolution_chart, use_container_width=True)
        
        st.markdown("### 📅 Time Decay Analysis")
        st.info("""
        **Key Insights:**
        - **Delta** typically decreases as expiration approaches (for OTM options)
        - **Gamma** peaks near expiration for ATM options
        - **Theta** accelerates as expiration approaches
        - **Vega** decreases steadily over time
        """)
    
    with tab3:
        scenario_chart = GreeksDashboard.create_scenario_analysis_chart(portfolio.positions)
        st.plotly_chart(scenario_chart, use_container_width=True)
        
        st.markdown("### 🎯 Stress Testing")
        st.info("""
        **Scenario Definitions:**
        - **Bull Market**: +10% price, -2% volatility
        - **Bear Market**: -10% price, +2% volatility  
        - **High Vol**: 0% price, +5% volatility
        - **Low Vol**: 0% price, -5% volatility
        - **Crisis**: -20% price, +10% volatility
        """)
    
    st.markdown("---")
    
    st.markdown("### ⚡ Real-Time Greeks Monitor")
    if len(portfolio.positions) > 0:
        net_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
        options_count = 0
        for position in portfolio.positions.values():
            if position.strategy == "Options":
                options_count += 1
                greeks = position.get_position_greeks()
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value
    
        if options_count > 0:
            col1, col2, col3, col4, col5 = st.columns(5)
        
            with col1:
                delta_color = "#4ECDC4" if net_greeks['delta'] >= 0 else "#FF6B6B"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: {delta_color};">Portfolio Delta</div>
                    <div class="metric-value">{net_greeks['delta']:.3f}</div>
                    <div class="metric-subtitle">Price Sensitivity</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                gamma_color = "#48bb78" if net_greeks['gamma'] >= 0 else "#e74c3c"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: {gamma_color};">Portfolio Gamma</div>
                    <div class="metric-value">{net_greeks['gamma']:.4f}</div>
                    <div class="metric-subtitle">Convexity</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                theta_color = "#FF6B6B" if net_greeks['theta'] < 0 else "#48bb78"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: {theta_color};">Portfolio Theta</div>
                    <div class="metric-value">${net_greeks['theta']:.2f}</div>
                    <div class="metric-subtitle">Daily Decay</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                vega_color = "#667eea"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: {vega_color};">Portfolio Vega</div>
                    <div class="metric-value">${net_greeks['vega']:.2f}</div>
                    <div class="metric-subtitle">Vol Sensitivity</div>
                </div>
                """, unsafe_allow_html=True)

            with col5:
                rho_color = "#8e44ad"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: {rho_color};">Portfolio Rho</div>
                    <div class="metric-value">${net_greeks['rho']:.2f}</div>
                    <div class="metric-subtitle">Rate Sensitivity</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 No options positions found. Add options positions to see Greeks analysis.")

if __name__ == "__main__":
    print("=== Greeks Dashboard Module ===")
    print("This module provides comprehensive Greeks visualization")
    print("for options portfolios in commodity hedging platforms.")
