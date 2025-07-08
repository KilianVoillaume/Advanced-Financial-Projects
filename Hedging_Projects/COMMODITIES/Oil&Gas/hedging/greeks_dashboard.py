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

# Import our options math module
from hedging.options_math import BlackScholesCalculator, get_risk_free_rate, get_commodity_volatility, time_to_expiration


class GreeksDashboard:
    """Updated Greeks dashboard with fixed calculation."""
    
    @staticmethod
    def _calculate_position_greeks(position) -> Dict[str, float]:
        """FIXED: Use the Position's own Greeks calculation."""
        return position.get_position_greeks()
    
    @staticmethod
    def render_greeks_summary_cards(portfolio_positions: Dict):
        """Enhanced summary cards with position type awareness."""
        
        # Calculate net portfolio Greeks
        net_greeks = {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0
        }
        
        options_count = 0
        underlying_count = 0
        hedge_count = 0
        speculation_count = 0
        
        for position in portfolio_positions.values():
            if position.strategy == "Options":
                options_count += 1
                
                # Count by position type
                if position.position_type == PositionType.UNDERLYING:
                    underlying_count += 1
                elif position.position_type == PositionType.HEDGE:
                    hedge_count += 1
                elif position.position_type == PositionType.SPECULATION:
                    speculation_count += 1
                
                greeks = position.get_position_greeks()
                for greek_name, greek_value in greeks.items():
                    net_greeks[greek_name] += greek_value
        
        if options_count == 0:
            st.info("📊 No options positions found. Add options positions to see Greeks analysis.")
            return
        
        # Display enhanced summary
        st.markdown(f"""
        **📊 Portfolio Summary:** {options_count} options positions 
        ({hedge_count} hedges, {speculation_count} speculative, {underlying_count} other)
        """)
        
        # Display summary cards (existing code...)
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
        
        # Risk interpretation
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
        """Render real-time Greeks monitoring interface."""
        
        st.markdown("**⚡ Real-Time Greeks Monitor**")
        
        # Create auto-refreshing display
        if st.button("🔄 Refresh Greeks", type="secondary"):
            st.rerun()
        
        # Create detailed Greeks table
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
        """Create scenario analysis for portfolio Greeks."""
        
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
        
        # Define scenarios
        scenarios = [
            {"name": "Base Case", "price_shock": 0.0, "vol_shock": 0.0},
            {"name": "Bull Market", "price_shock": 0.10, "vol_shock": -0.02},
            {"name": "Bear Market", "price_shock": -0.10, "vol_shock": 0.02},
            {"name": "High Vol", "price_shock": 0.0, "vol_shock": 0.05},
            {"name": "Low Vol", "price_shock": 0.0, "vol_shock": -0.05},
            {"name": "Crisis", "price_shock": -0.20, "vol_shock": 0.10}
        ]
        
        # Calculate scenario impacts
        scenario_results = []
        
        for scenario in scenarios:
            total_pnl = 0.0
            
            for position in portfolio_positions.values():
                if position.strategy == "Options":
                    greeks = GreeksDashboard._calculate_position_greeks(position)
                    
                    # Calculate P&L impact
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
        
        # Create scenario chart
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
    """
    Real-time Greeks monitoring and alerting system.
    """
    
    @staticmethod
    def check_risk_limits(portfolio_positions: Dict) -> Dict[str, List[str]]:
        """
        Check portfolio Greeks against predefined risk limits.
        
        Returns:
        --------
        Dict with risk alerts by category
        """
        alerts = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
        # Calculate net Greeks
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
        
        # Define risk limits
        limits = {
            'delta': {'critical': 1.0, 'warning': 0.5},
            'gamma': {'critical': 0.1, 'warning': 0.05},
            'theta': {'critical': -100, 'warning': -50},
            'vega': {'critical': 500, 'warning': 200}
        }
        
        # Check limits
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
        
        # Add positive alerts
        if len(alerts['critical']) == 0 and len(alerts['warning']) == 0:
            alerts['info'].append("All Greeks within acceptable risk limits")
        
        return alerts
    
    @staticmethod
    def render_risk_alerts(portfolio_positions: Dict):
        """Render risk alerts dashboard."""
        
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
        
        # Create risk gauge
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
        """Create chart showing how Greeks evolve over time."""
        
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
        
        # Calculate Greeks evolution
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
                    # Simulate time decay
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
        
        # Create evolution chart
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
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Days", row=2, col=1)
        fig.update_xaxes(title_text="Days", row=2, col=2)
        
        return fig


def render_enhanced_greeks_tab(portfolio, analysis_ready):
    """ Enhanced Greeks monitoring tab for the main application. """
    
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
        
        # Greeks heatmap
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
    
    # Real-time monitoring
    st.markdown("### ⚡ Real-Time Greeks Monitor")
    GreeksDashboard.render_real_time_greeks_monitor(portfolio.positions)


if __name__ == "__main__":
    print("=== Greeks Dashboard Module ===")
    print("This module provides comprehensive Greeks visualization")
    print("for options portfolios in commodity hedging platforms.")
