"""
app.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from hedging.data import get_prices, get_current_price, get_available_commodities
from hedging.strategies import compute_payoff_diagram, get_hedge_summary
from hedging.simulation import simulate_hedged_vs_unhedged, compare_hedging_effectiveness
from hedging.risk import calculate_risk_metrics, calculate_delta_exposure, summarize_risk_comparison
from hedging.stress_testing import STRESS_SCENARIOS
from hedging.portfolio import (
    PortfolioManager, Position, 
    create_oil_position, create_gas_position, create_brent_position,
    create_sample_portfolio
)

# Page configuration
st.set_page_config(
    page_title="Oil & Gas Hedging Platform",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-container { font-family: 'Inter', sans-serif; }
    
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem;
        text-align: center; color: white; box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3.5rem; font-weight: 700; margin-bottom: 0.8rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle { font-size: 1.3rem; font-weight: 300; opacity: 0.95; }
    
    .mode-container {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 2rem; border-radius: 20px; margin: 2rem auto; max-width: 800px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); text-align: center;
    }
    
    .mode-title {
        color: white; font-size: 1.8rem; font-weight: 600; margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .stRadio > div { display: flex; justify-content: center; gap: 2rem; }
    
    .stRadio > div > label {
        background: rgba(255,255,255,0.15); padding: 1rem 2rem; border-radius: 15px;
        backdrop-filter: blur(10px); border: 2px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease; color: white; font-weight: 500; font-size: 1.1rem;
        cursor: pointer; min-width: 250px; text-align: center;
    }
    
    .stRadio > div > label:hover {
        background: rgba(255,255,255,0.25); transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: rgba(255,255,255,0.9); color: #2d3748; font-weight: 600;
        border-color: rgba(255,255,255,0.8);
    }
    
    .portfolio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 20px; color: white; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); transition: transform 0.3s ease;
    }
    
    .portfolio-card:hover { transform: translateY(-5px); }
    
    .position-card {
        background: white; padding: 1.5rem; border-radius: 15px;
        border-left: 5px solid #667eea; margin-bottom: 1rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08); transition: all 0.3s ease;
    }
    
    .position-card:hover {
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); transform: translateY(-2px);
    }
    
    .metric-card {
        background: white; padding: 1.8rem; border-radius: 15px; margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08); text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #2d3748; margin-bottom: 0.3rem;
    }
    
    .metric-subtitle { font-size: 0.8rem; color: #666; }
    
    .sidebar-section {
        background: linear-gradient(145deg, #f8f9ff 0%, #e8f4fd 100%);
        padding: 1.8rem; border-radius: 15px; margin-bottom: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }
    
    .sidebar-title {
        color: #2d3748; font-weight: 600; font-size: 1.2rem; margin-bottom: 1.5rem;
        display: flex; align-items: center; gap: 0.8rem;
    }
    
    .success-banner {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3); font-weight: 500;
        text-align: center; font-size: 1.1rem;
    }
    
    .section-header {
        font-size: 2rem; font-weight: 600; color: #2d3748; margin: 2.5rem 0 1.5rem 0;
        display: flex; align-items: center; gap: 0.8rem;
    }
    
    .stButton > button {
        border-radius: 12px; border: none; padding: 0.8rem 1.5rem;
        font-weight: 600; font-size: 1rem; transition: all 0.3s ease; width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); color: white;
    }
    
    .stSelectbox > div > div { border-radius: 10px; border: 2px solid #e2e8f0; }
    .stTextInput > div > div > input { border-radius: 10px; border: 2px solid #e2e8f0; padding: 0.8rem; }
    .stNumberInput > div > div > input { border-radius: 10px; border: 2px solid #e2e8f0; padding: 0.8rem; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; margin-bottom: 2rem; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; border-radius: 15px; padding: 0 30px; background: #f7fafc;
        border: 2px solid transparent; font-weight: 500; font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-color: rgba(255,255,255,0.3);
    }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in { animation: fadeIn 0.8s ease-out; }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2196f3;
        margin: 1rem 0; font-weight: 500;
    }
    
    .empty-state {
        text-align: center; padding: 4rem 2rem;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 20px; margin: 2rem 0;
    }
    
    .empty-state-icon { font-size: 4rem; margin-bottom: 1.5rem; }
    .empty-state-title { font-size: 2rem; color: #2d3748; margin-bottom: 1rem; font-weight: 600; }
    .empty-state-text { color: #718096; font-size: 1.2rem; margin-bottom: 2rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'portfolio_mode' not in st.session_state:
        st.session_state.portfolio_mode = True
    if 'portfolio_manager' not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    if 'single_position_results' not in st.session_state:
        st.session_state.single_position_results = None


def main():
    """Main application function."""
    initialize_session_state()
    
    st.markdown("""
    <div class="hero-header fade-in">
        <div class="hero-title">🛢️ Oil & Gas Hedging Platform</div>
        <div class="hero-subtitle">Professional multi-commodity risk management and portfolio optimization</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mode-container">
        <div class="mode-title">Choose Your Analysis Mode</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        mode_option = st.radio(
            "",
            options=["🎯 Single Position Analysis", "📊 Portfolio Management"],
            index=1 if st.session_state.portfolio_mode else 0,
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.portfolio_mode = (mode_option == "📊 Portfolio Management")
    
    if st.session_state.portfolio_mode:
        st.markdown("""
        <div class="info-box">
            📊 <strong>Portfolio Mode</strong>: Multi-commodity portfolio analysis with correlation matrices, 
            advanced risk metrics, and optimization algorithms
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            🎯 <strong>Single Position Mode</strong>: Deep-dive analysis of individual commodity positions 
            with detailed payoff diagrams and Monte Carlo simulations
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.portfolio_mode:
        portfolio_interface()
    else:
        single_position_interface()


def portfolio_interface():
    """Portfolio management interface."""
    st.markdown("""
    <div class="success-banner fade-in">
        ✨ <strong>Portfolio Mode Active</strong> - Advanced multi-commodity risk management platform
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 2])
    
    with col1:
        portfolio_builder_sidebar()
    with col2:
        portfolio_dashboard()


def portfolio_builder_sidebar():
    """Portfolio builder sidebar."""
    st.markdown('<div class="section-header">🏗️ Portfolio Builder</div>', unsafe_allow_html=True)
    
    portfolio = st.session_state.portfolio_manager
    
    if len(portfolio) > 0:
        total_notional = sum(pos.notional_value for pos in portfolio.positions.values())
        unique_commodities = len(set(pos.commodity for pos in portfolio.positions.values()))
        
        st.markdown(f"""
        <div class="portfolio-card fade-in">
            <h3 style="margin-bottom: 1.5rem; font-size: 1.4rem;">📊 Portfolio Overview</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.3rem;">{len(portfolio)}</div>
                    <div style="opacity: 0.9; font-size: 0.9rem;">Positions</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.3rem;">{unique_commodities}</div>
                    <div style="opacity: 0.9; font-size: 0.9rem;">Commodities</div>
                </div>
            </div>
            <div style="text-align: center; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
                <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.3rem;">${total_notional:,.0f}</div>
                <div style="opacity: 0.9; font-size: 0.9rem;">Total Notional Value</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🚀 Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Load Sample", type="secondary", use_container_width=True):
            st.session_state.portfolio_manager = create_sample_portfolio()
            st.success("✅ Sample portfolio loaded!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            st.session_state.portfolio_manager.clear()
            st.success("✅ Portfolio cleared!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add new position form
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">➕ Add New Position</div>', unsafe_allow_html=True)
    
    with st.form("add_position_form", clear_on_submit=True):
        position_name = st.text_input(
            "Position Name:",
            placeholder="e.g., 'oil_main', 'gas_hedge'",
            help="Unique identifier for this position"
        )
        
        commodity = st.selectbox(
            "Commodity:",
            options=["WTI Crude Oil", "Brent Crude Oil", "Natural Gas"],
            help="Select commodity type"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            position_direction = st.selectbox(
                "Direction:",
                options=["Long", "Short"],
                help="Position direction"
            )
        
        with col2:
            position_size = st.number_input(
                "Size:",
                min_value=1.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0,
                help="Position size (barrels/MMBtu)"
            )
        
        if position_direction == "Short":
            position_size = -position_size
        
        hedge_ratio = st.slider(
            "Hedge Ratio:",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            format="%.0f%%",
            help="Percentage of position to hedge"
        ) 
        
        st.caption(f"Hedging {hedge_ratio*100:.0f}% of the position")
        
        strategy = st.selectbox(
            "Strategy:",
            options=["Futures", "Options"],
            help="Hedging instrument"
        )
        
        strike_price = None
        option_expiry = None
        
        if strategy == "Options":
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    current_price = get_current_price(commodity)
                    strike_price = st.slider(
                        "Strike Price:",
                        min_value=float(current_price * 0.7),
                        max_value=float(current_price * 1.3),
                        value=float(current_price),
                        step=0.5,
                        help="Option strike price"
                    )
                    
                    moneyness = current_price / strike_price
                    if abs(moneyness - 1.0) < 0.05:
                        st.caption("🎯 At-the-Money (ATM)")
                    elif moneyness > 1.05:
                        st.caption("📉 Out-of-the-Money (OTM)")
                    else:
                        st.caption("📈 In-the-Money (ITM)")
                        
                except:
                    strike_price = st.number_input("Strike Price:", value=75.0)
            
            with col2:
                option_expiry = st.selectbox(
                    "Maturity:",
                    options=[1, 3, 6, 12],
                    index=1,
                    format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
                    help="Time until option expiration"
                )
        
        submitted = st.form_submit_button("🔥 Add Position", type="primary", use_container_width=True)
        
        if submitted and position_name:
            if position_name not in st.session_state.portfolio_manager.positions:
                new_position = Position(
                    commodity=commodity,
                    size=position_size,
                    hedge_ratio=hedge_ratio,
                    strategy=strategy,
                    strike_price=strike_price
                )
                
                st.session_state.portfolio_manager.add_position(position_name, new_position)
                st.success(f"✅ Added {position_name} to portfolio!")
                st.rerun()
            else:
                st.error(f"❌ Position '{position_name}' already exists!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current positions
    if len(portfolio) > 0:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📋 Current Positions</div>', unsafe_allow_html=True)
        
        for name, position in portfolio.positions.items():
            direction_emoji = "📈" if position.size > 0 else "📉"
            direction_text = "Long" if position.size > 0 else "Short"
            hedge_status = "🛡️" if position.hedge_ratio > 0.5 else "⚠️" if position.hedge_ratio > 0 else "🚫"
            
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"""
                <div class="position-card">
                    <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.3rem;">
                        {direction_emoji} {name}
                    </div>
                    <div style="color: #666; font-size: 0.9rem; line-height: 1.4;">
                        {position.commodity}<br>
                        {direction_text} {abs(position.size):,.0f} • {position.hedge_ratio*100:.0f}% hedged {hedge_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("❌", key=f"remove_{name}", help=f"Remove {name}"):
                    st.session_state.portfolio_manager.remove_position(name)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚙️ Analysis Settings</div>', unsafe_allow_html=True)
    
    confidence_level = st.slider(
        "Confidence Level:",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        format="%d%%"
    ) / 100.0
    
    n_simulations = st.selectbox(
        "Monte Carlo Runs:",
        options=[1000, 5000, 10000],
        index=1,
        format_func=lambda x: f"{x:,} simulations"
    )
    
    st.session_state.portfolio_manager.set_config(
        confidence_level=confidence_level,
        simulation_runs=n_simulations
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Run Portfolio Analysis", type="primary", use_container_width=True, disabled=len(portfolio) == 0):
        if len(portfolio) > 0:
            st.session_state.simulation_run = True
            st.rerun()


def portfolio_dashboard():
    """Portfolio dashboard."""
    portfolio = st.session_state.portfolio_manager
    
    if len(portfolio) == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🎯</div>
            <div class="empty-state-title">Ready to Build Your Portfolio?</div>
            <div class="empty-state-text">
                Start by adding commodity positions using the builder on the left.<br>
                Create long/short positions, set hedge ratios, and analyze risk metrics.
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; display: inline-block; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                💡 <strong>Tip:</strong> Try the "Load Sample" button for a quick start with a diversified portfolio!
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    with st.spinner("🔄 Calculating portfolio analytics..."):
        try:
            portfolio.calculate_correlations().calculate_portfolio_risk()
            analysis_ready = True
        except Exception as e:
            st.error(f"Error in analytics: {e}")
            analysis_ready = False
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🔗 Correlations", 
        "⚠️ Risk Analysis", 
        "🧪 Stress Testing"
    ])
    
    with tab1:
        portfolio_overview_tab(portfolio, analysis_ready)
    with tab2:
        correlations_tab(portfolio, analysis_ready)
    with tab3:
        risk_analysis_tab(portfolio, analysis_ready)
    with tab4:
        stress_testing_tab(portfolio, analysis_ready)


def portfolio_overview_tab(portfolio, analysis_ready):
    """Portfolio overview tab."""
    st.markdown('<div class="section-header">📊 Portfolio Composition</div>', unsafe_allow_html=True)
    
    if analysis_ready:
        risk_summary = portfolio.get_portfolio_risk_summary()
        if risk_summary:
            col1, col2, col3, col4 = st.columns(4)
            metrics = list(risk_summary.items())
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #4ECDC4;">Expected P&L</div>
                    <div class="metric-value">{metrics[0][1]}</div>
                    <div class="metric-subtitle">Expected Return</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #FF6B6B;">VaR (95%)</div>
                    <div class="metric-value">{metrics[1][1]}</div>
                    <div class="metric-subtitle">Value at Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #667eea;">Volatility</div>
                    <div class="metric-value">{metrics[3][1]}</div>
                    <div class="metric-subtitle">Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #48bb78;">Sharpe Ratio</div>
                    <div class="metric-value">{metrics[4][1]}</div>
                    <div class="metric-subtitle">Risk-Adj. Return</div>
                </div>
                """, unsafe_allow_html=True)
            
            if len(metrics) > 5:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #dc3545;">CVaR (95%)</div>
                        <div class="metric-value">{metrics[2][1]}</div>
                        <div class="metric-subtitle">Tail Risk</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #8e44ad;">Max Drawdown</div>
                        <div class="metric-value">{metrics[5][1]}</div>
                        <div class="metric-subtitle">Worst Loss</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #2c3e50;">Positions</div>
                        <div class="metric-value">{metrics[6][1]}</div>
                        <div class="metric-subtitle">Total Count</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #16a085;">Total Notional</div>
                        <div class="metric-value">{metrics[7][1]}</div>
                        <div class="metric-subtitle">Portfolio Size</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Portfolio composition
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        weights = portfolio.get_portfolio_weights()
        if weights:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.5,
                textinfo='label+percent',
                textposition='auto',
                marker=dict(
                    colors=['#667eea', '#764ba2', '#4ECDC4', '#FF6B6B', '#48bb78', '#f39c12', '#e74c3c'],
                    line=dict(color='white', width=3)
                )
            )])
            
            fig_pie.update_layout(
                title={
                    'text': "Portfolio Allocation by Position",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Inter', 'color': '#2d3748'}
                },
                showlegend=True,
                height=450,
                margin=dict(t=80, b=40, l=40, r=40),
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(size=12)
                ),
                font=dict(family="Inter")
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        summary_df = portfolio.get_portfolio_summary()
        if not summary_df.empty:
            st.markdown("**📋 Position Details:**")
            display_df = summary_df[['Position Name', 'Commodity', 'Direction', 'Weight', 'Hedge Ratio']].copy()
            st.dataframe(
                display_df, 
                use_container_width=True, 
                hide_index=True,
                height=320
            )
        
        exposure_df = portfolio.get_commodity_exposure()
        if not exposure_df.empty:
            st.markdown("**🎯 Net Commodity Exposure:**")
            st.dataframe(exposure_df, use_container_width=True, hide_index=True, height=150)


def correlations_tab(portfolio, analysis_ready):
    """Correlations analysis tab."""
    st.markdown('<div class="section-header">🔗 Cross-Commodity Correlations</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to calculate correlations")
        return
    
    corr_matrix = portfolio.get_correlation_matrix()
    
    if corr_matrix.empty:
        st.info("ℹ️ Add positions with different commodities to see correlations")
        return
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white", "family": "Inter"},
        colorbar=dict(title="Correlation")
    ))
    
    fig_heatmap.update_layout(
        title={
            'text': "Commodity Correlation Matrix",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Inter', 'color': '#2d3748'}
        },
        height=500,
        xaxis_title="Commodities",
        yaxis_title="Commodities",
        font=dict(family="Inter", size=14)
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("### 📋 Correlation Analysis")
    
    correlations = []
    commodities = corr_matrix.columns.tolist()
    
    for i, commodity1 in enumerate(commodities):
        for j, commodity2 in enumerate(commodities):
            if i < j:
                corr_value = corr_matrix.loc[commodity1, commodity2]
                correlations.append({
                    'Commodity Pair': f"{commodity1} ↔ {commodity2}",
                    'Correlation': f"{corr_value:.3f}",
                    'Strength': get_correlation_strength(corr_value),
                    'Interpretation': get_correlation_interpretation(corr_value)
                })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        st.dataframe(corr_df, use_container_width=True, hide_index=True, height=300)


def risk_analysis_tab(portfolio, analysis_ready):
    """Risk analysis tab."""
    st.markdown('<div class="section-header">⚠️ Portfolio Risk Analysis</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to see risk metrics")
        return
    
    try:
        portfolio_pnl = portfolio._simulate_portfolio_pnl()
        
        if len(portfolio_pnl) > 0:
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=portfolio_pnl,
                nbinsx=60,
                name='Portfolio P&L',
                marker=dict(
                    color='rgba(102, 126, 234, 0.7)',
                    line=dict(color='rgba(102, 126, 234, 1)', width=1)
                ),
                opacity=0.8
            ))
            
            var_95 = np.percentile(portfolio_pnl, 5)
            cvar_95 = np.mean(portfolio_pnl[portfolio_pnl <= var_95])
            
            fig_dist.add_vline(
                x=var_95,
                line_dash="dash",
                line_color="#FF6B6B",
                line_width=4,
                annotation_text=f"VaR (95%): ${var_95:,.0f}",
                annotation_position="top"
            )
            
            fig_dist.add_vline(
                x=cvar_95,
                line_dash="dot",
                line_color="#dc3545",
                line_width=4,
                annotation_text=f"CVaR (95%): ${cvar_95:,.0f}",
                annotation_position="top"
            )
            
            fig_dist.update_layout(
                title={
                    'text': "Portfolio P&L Distribution (Monte Carlo Simulation)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Inter', 'color': '#2d3748'}
                },
                xaxis_title="P&L ($)",
                yaxis_title="Frequency",
                height=450,
                showlegend=False,
                font=dict(family="Inter", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.markdown("### 📊 Key Risk Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #FF6B6B;">Value at Risk</div>
                    <div class="metric-value">${var_95:,.0f}</div>
                    <div class="metric-subtitle">95% Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #dc3545;">Conditional VaR</div>
                    <div class="metric-value">${cvar_95:,.0f}</div>
                    <div class="metric-subtitle">Expected Tail Loss</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                prob_loss = np.sum(portfolio_pnl < 0) / len(portfolio_pnl) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #667eea;">Loss Probability</div>
                    <div class="metric-value">{prob_loss:.1f}%</div>
                    <div class="metric-subtitle">Chance of Loss</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                volatility = np.std(portfolio_pnl)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #4ECDC4;">Volatility</div>
                    <div class="metric-value">${volatility:,.0f}</div>
                    <div class="metric-subtitle">Standard Deviation</div>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                expected_return = np.mean(portfolio_pnl)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #48bb78;">Expected Return</div>
                    <div class="metric-value">${expected_return:,.0f}</div>
                    <div class="metric-subtitle">Mean P&L</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #8e44ad;">Sharpe Ratio</div>
                    <div class="metric-value">{sharpe_ratio:.3f}</div>
                    <div class="metric-subtitle">Risk-Adj. Return</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                max_gain = np.max(portfolio_pnl)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #16a085;">Max Gain</div>
                    <div class="metric-value">${max_gain:,.0f}</div>
                    <div class="metric-subtitle">Best Case</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                skewness = float(pd.Series(portfolio_pnl).skew())
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #f39c12;">Skewness</div>
                    <div class="metric-value">{skewness:.2f}</div>
                    <div class="metric-subtitle">Distribution Shape</div>
                </div>
                """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Could not generate risk analysis: {e}")


def stress_testing_tab(portfolio, analysis_ready):
    """Stress testing tab."""
    st.markdown('<div class="section-header">🧪 Historical Crisis Stress Testing</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to enable stress testing")
        return
    
    st.markdown("""
    **Test your portfolio against major historical market crises** to understand how it would have 
    performed during extreme market conditions and validate your hedging strategy.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_scenarios = st.multiselect(
            "📊 **Select Crisis Scenarios:**",
            options=list(STRESS_SCENARIOS.keys()),
            default=list(STRESS_SCENARIOS.keys())[:3],
            help="Choose historical crises to simulate"
        )
    
    with col2:
        if st.button("🚀 Run All Scenarios", type="primary", use_container_width=True):
            selected_scenarios = list(STRESS_SCENARIOS.keys())
    
    if selected_scenarios:
        stress_results = []
        
        for scenario_name in selected_scenarios:
            scenario = STRESS_SCENARIOS[scenario_name]
            
            total_unhedged = 0
            total_hedged = 0
            
            for name, position in portfolio.positions.items():
                commodity_type = "gas" if "gas" in position.commodity.lower() else "oil"
                
                price_shock = scenario["oil_peak_to_trough"]
                if commodity_type == "gas":
                    price_shock *= scenario["gas_correlation"]
                
                price_change = position.current_price * price_shock
                unhedged_pnl = price_change * position.size
                
                if position.strategy == "Futures":
                    hedge_pnl = -price_change * position.size * position.hedge_ratio
                elif position.strategy == "Options":
                    if price_change < 0:
                        hedge_pnl = -price_change * position.size * position.hedge_ratio * 0.8
                    else:
                        hedge_pnl = -price_change * position.size * position.hedge_ratio * 0.2
                else:
                    hedge_pnl = 0
                
                total_unhedged += unhedged_pnl
                total_hedged += unhedged_pnl + hedge_pnl
            
            hedge_benefit = total_hedged - total_unhedged
            protection = abs(hedge_benefit) / abs(total_unhedged) * 100 if total_unhedged != 0 else 0
            
            stress_results.append({
                "Crisis": scenario_name,
                "Timeline": scenario["timeline"],
                "Price Impact": f"{scenario['oil_peak_to_trough']:.1%}",
                "Unhedged P&L": f"${total_unhedged:,.0f}",
                "Hedged P&L": f"${total_hedged:,.0f}",
                "Hedge Benefit": f"${hedge_benefit:,.0f}",
                "Protection": f"{protection:.1f}%"
            })
        
        st.markdown("### 📊 Stress Test Results")
        results_df = pd.DataFrame(stress_results)
        st.dataframe(results_df, use_container_width=True, hide_index=True, height=300)
        
        if stress_results:
            unhedged_values = [float(r["Unhedged P&L"].replace("$", "").replace(",", "")) for r in stress_results]
            hedged_values = [float(r["Hedged P&L"].replace("$", "").replace(",", "")) for r in stress_results]
            
            worst_unhedged = min(unhedged_values)
            worst_hedged = min(hedged_values)
            avg_protection = np.mean([float(r["Protection"].replace("%", "")) for r in stress_results])
            
            st.markdown("### 🎯 Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #FF6B6B;">Worst Unhedged Loss</div>
                    <div class="metric-value">${worst_unhedged:,.0f}</div>
                    <div class="metric-subtitle">Maximum Exposure</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #4ECDC4;">Worst Hedged Loss</div>
                    <div class="metric-value">${worst_hedged:,.0f}</div>
                    <div class="metric-subtitle">Protected Loss</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_reduction = (abs(worst_unhedged) - abs(worst_hedged)) / abs(worst_unhedged) * 100 if worst_unhedged != 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #48bb78;">Risk Reduction</div>
                    <div class="metric-value">{risk_reduction:.1f}%</div>
                    <div class="metric-subtitle">Hedge Effectiveness</div>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Select crisis scenarios above to run stress tests")


def get_correlation_strength(correlation):
    """Get correlation strength description."""
    abs_corr = abs(correlation)
    if abs_corr > 0.7:
        return "🔴 Strong"
    elif abs_corr > 0.3:
        return "🟡 Moderate"
    else:
        return "🟢 Weak"


def get_correlation_interpretation(correlation):
    """Get correlation interpretation."""
    if correlation > 0.7:
        return "Strong positive relationship - prices move together"
    elif correlation > 0.3:
        return "Moderate positive relationship"
    elif correlation > -0.3:
        return "Little to no linear relationship"
    elif correlation > -0.7:
        return "Moderate negative relationship"
    else:
        return "Strong negative relationship - prices move opposite"


def single_position_interface():
    """Complete single position interface."""
    st.markdown('<div class="section-header">🎯 Single Position Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🏭 Commodity Selection</div>', unsafe_allow_html=True)
        
        commodity = st.selectbox(
            "Select Commodity:",
            options=["WTI Crude Oil", "Brent Crude Oil", "Natural Gas"],
            index=0,
            help="Select the commodity for hedging analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📈 Position Parameters</div>', unsafe_allow_html=True)
        
        position_size = st.number_input(
            "Position Size:",
            min_value=1.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
            help="Position size (barrels for oil, MMBtu for natural gas)"
        )
        
        position_type = st.radio(
            "Position Type:",
            options=["Long", "Short"],
            index=0,
            help="Long = expecting price increase, Short = expecting price decrease"
        )
        
        if position_type == "Short":
            position_size = -position_size
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🛡️ Hedging Strategy</div>', unsafe_allow_html=True)
        
        strategy = st.selectbox(
            "Hedging Strategy:",
            options=["Futures", "Options"],
            index=0,
            help="Choose hedging instrument"
        )
        
        hedge_ratio = st.slider(
            "Hedge Ratio:",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            format="%.0f%%",
            help="Percentage of position to hedge"
        )
        
        st.caption(f"Hedging {hedge_ratio*100:.0f}% of the position")
        st.markdown('</div>', unsafe_allow_html=True)
        
        strike_price = None
        option_expiry = None
        
        if strategy == "Options":
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">⚙️ Options Parameters</div>', unsafe_allow_html=True)
            
            try:
                current_price = get_current_price(commodity)
                
                strike_price = st.slider(
                    "Strike Price ($):",
                    min_value=float(current_price * 0.7),
                    max_value=float(current_price * 1.3),
                    value=float(current_price),
                    step=0.5,
                    help="Option strike price"
                )
                
                moneyness = float(current_price) / float(strike_price)
                if abs(moneyness - 1.0) < 0.05:
                    st.caption("🎯 At-the-Money (ATM)")
                elif moneyness > 1.05:
                    st.caption("📉 Out-of-the-Money (OTM)")
                else:
                    st.caption("📈 In-the-Money (ITM)")
                
                option_expiry = st.selectbox(
                    "Option Expiration:",
                    options=[1, 3, 6, 12],
                    index=1,
                    format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
                    help="Time until option expiration"
                )
                
            except Exception as e:
                st.error(f"Error getting current price: {e}")
                strike_price = 75.0
                option_expiry = 3
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 Risk Analysis</div>', unsafe_allow_html=True)
        
        confidence = st.slider(
            "Confidence Level:",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            format="%d%%"
        ) / 100.0
        
        n_simulations = st.selectbox(
            "Number of Simulations:",
            options=[1000, 5000, 10000],
            index=1,
            format_func=lambda x: f"{x:,} simulations"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state.simulation_run = True
            st.rerun()
    
    # Main content area
    if st.session_state.simulation_run:
        with st.spinner("🔄 Loading price data and running simulation..."):
            try:
                prices = get_prices(commodity)
                current_price = float(get_current_price(commodity))
                
                if strategy == "Options" and strike_price is not None:
                    strike_price = float(strike_price)
                
                sim_results = simulate_hedged_vs_unhedged(
                    prices, position_size, hedge_ratio, strategy, strike_price, n_simulations
                )
                
                payoff_data = compute_payoff_diagram(
                    float(current_price), position_size, hedge_ratio, strategy, 
                    float(strike_price) if strike_price is not None else None
                )
                
                hedged_risk = calculate_risk_metrics(sim_results['hedged_pnl'], confidence)
                unhedged_risk = calculate_risk_metrics(sim_results['unhedged_pnl'], confidence)
                
                st.session_state.single_position_results = {
                    'prices': prices,
                    'current_price': current_price,
                    'sim_results': sim_results,
                    'payoff_data': payoff_data,
                    'hedged_risk': hedged_risk,
                    'unhedged_risk': unhedged_risk,
                    'params': {
                        'commodity': commodity,
                        'position': position_size,
                        'strategy': strategy,
                        'hedge_ratio': hedge_ratio,
                        'strike_price': strike_price,
                        'confidence': confidence
                    }
                }
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")
                st.session_state.simulation_run = False
    
    # Display results if available
    if st.session_state.single_position_results:
        display_single_position_results()
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🎯</div>
            <div class="empty-state-title">Single Position Analysis</div>
            <div class="empty-state-text">
                Configure your position parameters in the sidebar and click "Run Analysis" 
                to see detailed payoff diagrams, risk metrics, and Monte Carlo simulations.
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; display: inline-block; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                💡 <strong>Features:</strong> Options & Futures analysis, VaR calculations, stress testing, payoff diagrams
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_single_position_results():
    """Display single position analysis results."""
    results = st.session_state.single_position_results
    params = results['params']
    
    st.markdown('<div class="section-header">📋 Analysis Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Commodity</div>
            <div class="metric-value" style="font-size: 1.5rem;">{params['commodity']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current Price</div>
            <div class="metric-value">${results['current_price']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        direction = "Long" if params['position'] > 0 else "Short"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Position</div>
            <div class="metric-value" style="font-size: 1.3rem;">{direction}</div>
            <div class="metric-subtitle">{abs(params['position']):,.0f} units</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Hedge Strategy</div>
            <div class="metric-value" style="font-size: 1.3rem;">{params['strategy']}</div>
            <div class="metric-subtitle">{params['hedge_ratio']*100:.0f}% hedged</div>
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "📈 Charts", "⚠️ Risk Metrics"])
    
    with tab1:
        st.markdown("### 🎯 Hedging Effectiveness")
        effectiveness = compare_hedging_effectiveness(
            results['sim_results']['hedged_pnl'], 
            results['sim_results']['unhedged_pnl']
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: #4ECDC4;">Volatility Reduction</div>
                <div class="metric-value">{effectiveness['volatility_reduction']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: #48bb78;">Loss Prob. Reduction</div>
                <div class="metric-value">{effectiveness['loss_prob_reduction']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: #667eea;">Expected P&L Change</div>
                <div class="metric-value">${effectiveness['mean_difference']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: #8e44ad;">Sharpe Improvement</div>
                <div class="metric-value">{effectiveness['sharpe_improvement']:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📈 Price Chart")
        
        prices = results['prices']
        current_price = results['current_price']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name=f'{params["commodity"]} Price',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="#FF6B6B",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        if params['strategy'] == 'Options' and params['strike_price']:
            fig.add_hline(
                y=params['strike_price'],
                line_dash="dot",
                line_color="#48bb78",
                annotation_text=f"Strike: ${params['strike_price']:.2f}"
            )
        
        fig.update_layout(
            title="Historical Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Payoff diagram
        st.markdown("### 📊 Payoff Diagram")
        
        payoff_data = results['payoff_data']
        
        fig_payoff = go.Figure()
        
        fig_payoff.add_trace(go.Scatter(
            x=payoff_data['spot_prices'],
            y=payoff_data['underlying_pnl'],
            mode='lines',
            name='Underlying P&L',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        fig_payoff.add_trace(go.Scatter(
            x=payoff_data['spot_prices'],
            y=payoff_data['hedge_pnl'],
            mode='lines',
            name='Hedge P&L',
            line=dict(color='#4ECDC4', width=2, dash='dot')
        ))
        
        fig_payoff.add_trace(go.Scatter(
            x=payoff_data['spot_prices'],
            y=payoff_data['net_pnl'],
            mode='lines',
            name='Net P&L',
            line=dict(color='#48bb78', width=3)
        ))
        
        fig_payoff.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig_payoff.add_vline(x=current_price, line_dash="dash", line_color="orange")
        
        for breakeven in payoff_data['breakeven_prices']:
            fig_payoff.add_vline(
                x=breakeven,
                line_dash="dot",
                line_color="purple",
                annotation_text=f"Breakeven: ${breakeven:.2f}"
            )
        
        fig_payoff.update_layout(
            title="Payoff Diagram at Expiry",
            xaxis_title="Spot Price ($)",
            yaxis_title="P&L ($)",
            height=400,
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig_payoff, use_container_width=True)
        
        # P&L Distribution
        st.markdown("### 📊 P&L Distribution")
        
        sim_results = results['sim_results']
        
        fig_hist = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Unhedged Position', 'Hedged Position'),
            shared_yaxes=True
        )
        
        fig_hist.add_trace(
            go.Histogram(
                x=sim_results['unhedged_pnl'],
                name='Unhedged P&L',
                opacity=0.7,
                marker_color='#FF6B6B',
                nbinsx=50
            ),
            row=1, col=1
        )
        
        fig_hist.add_trace(
            go.Histogram(
                x=sim_results['hedged_pnl'],
                name='Hedged P&L',
                opacity=0.7,
                marker_color='#4ECDC4',
                nbinsx=50
            ),
            row=1, col=2
        )
        
        fig_hist.update_layout(
            title="P&L Distribution Comparison",
            height=400,
            showlegend=False,
            font=dict(family="Inter")
        )
        
        fig_hist.update_xaxes(title_text="P&L ($)")
        fig_hist.update_yaxes(title_text="Frequency")
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.markdown("### ⚠️ Risk Comparison")
        
        # Risk comparison table
        risk_comparison = summarize_risk_comparison(results['hedged_risk'], results['unhedged_risk'])
        
        # Format the comparison table
        risk_comparison['Value_Unhedged'] = risk_comparison['Value_Unhedged'].apply(lambda x: f"${x:,.0f}")
        risk_comparison['Value_Hedged'] = risk_comparison['Value_Hedged'].apply(lambda x: f"${x:,.0f}")
        risk_comparison['Difference'] = risk_comparison['Difference'].apply(lambda x: f"${x:,.0f}")
        risk_comparison['Improvement'] = risk_comparison['Improvement'].apply(lambda x: f"{x:.1%}")
        
        risk_comparison.columns = ['Risk Metric', 'Unhedged', 'Hedged', 'Difference', 'Improvement']
        
        st.dataframe(risk_comparison, use_container_width=True, hide_index=True, height=300)
        
        # Additional metrics display
        st.markdown("### 📊 Additional Risk Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_exp = float(calculate_delta_exposure(
                results['prices'], params['position'], params['hedge_ratio'], 
                params['strategy'], params['strike_price']
            ))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: #667eea;">Delta Exposure</div>
                <div class="metric-value">{delta_exp:,.0f}</div>
                <div class="metric-subtitle">Net delta exposure</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            hedge_ratio = float(params['hedge_ratio'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title" style="color: #48bb78;">Hedge Effectiveness</div>
                <div class="metric-value">{hedge_ratio:.1%}</div>
                <div class="metric-subtitle">Position hedged</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Get Sharpe ratios
            hedged_sharpe_row = results['hedged_risk'][results['hedged_risk']['Metric'] == 'Sharpe Ratio']
            
            if not hedged_sharpe_row.empty:
                hedged_sharpe = float(hedged_sharpe_row['Value'].iloc[0])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #8e44ad;">Sharpe Ratio</div>
                    <div class="metric-value">{hedged_sharpe:.3f}</div>
                    <div class="metric-subtitle">Risk-adjusted return</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #8e44ad;">Sharpe Ratio</div>
                    <div class="metric-value">N/A</div>
                    <div class="metric-subtitle">Risk-adjusted return</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Stress testing for single position
        st.markdown("### 🧪 Crisis Stress Testing")
        
        selected_scenarios = st.multiselect(
            "Select crisis scenarios to test:",
            options=list(STRESS_SCENARIOS.keys()),
            default=list(STRESS_SCENARIOS.keys())[:3],
            key="single_position_stress"
        )
        
        if selected_scenarios:
            stress_results = []
            
            for scenario_name in selected_scenarios:
                scenario = STRESS_SCENARIOS[scenario_name]
                price_change = scenario["oil_peak_to_trough"]
                
                # Calculate position P&L
                position_size = abs(params['position'])
                price_shock = current_price * price_change
                
                # Unhedged P&L
                unhedged_pnl = price_shock * position_size
                
                # Hedged P&L
                hedge_ratio = params['hedge_ratio']
                if params['strategy'] == "Futures":
                    hedge_pnl = -price_shock * position_size * hedge_ratio
                elif params['strategy'] == "Options":
                    if price_change < 0:  # Downside protection
                        hedge_pnl = -price_shock * position_size * hedge_ratio * 0.8
                    else:  # Limited upside protection
                        hedge_pnl = -price_shock * position_size * hedge_ratio * 0.2
                else:
                    hedge_pnl = 0
                
                hedged_pnl = unhedged_pnl + hedge_pnl
                
                stress_results.append({
                    "Scenario": scenario_name,
                    "Timeline": scenario["timeline"],
                    "Price Change": f"{price_change:.1%}",
                    "Unhedged P&L": f"${unhedged_pnl:,.0f}",
                    "Hedged P&L": f"${hedged_pnl:,.0f}",
                    "Hedge Benefit": f"${hedge_pnl:,.0f}",
                    "Protection": f"{abs(hedge_pnl) / abs(unhedged_pnl):.1%}" if unhedged_pnl != 0 else "N/A"
                })
            
            # Display stress test results
            st.dataframe(pd.DataFrame(stress_results), use_container_width=True, hide_index=True)


# Call main function when script is run
if __name__ == "__main__":
    main()
