"""
app.py - Improved Oil & Gas Hedging Platform

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
from hedging.portfolio import (PortfolioManager, Position, create_oil_position, create_gas_position, create_brent_position, create_sample_portfolio)
from hedging.greeks_dashboard import GreeksDashboard, GreeksMonitor, render_enhanced_greeks_tab


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
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); border-radius: 20px; margin: 3rem auto; max-width: 1000px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); text-align: center; display: flex; justify-content: center;
        align-items: center; padding: 2rem;
    }
    
    .mode-title {
        color: white; font-size: 1.8rem; font-weight: 600; margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .stRadio > div { 
        display: flex; justify-content: center; gap: 3rem; 
        flex-wrap: wrap; align-items: center;
        margin: 0 auto;
        max-width: 800px;
    }
    
    .stRadio > div > label {
        background: rgba(255,255,255,0.15); padding: 1.5rem 3rem; border-radius: 15px;
        backdrop-filter: blur(10px); border: 2px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease; color: white; font-weight: 600; font-size: 1.2rem;
        cursor: pointer; min-width: 300px; text-align: center;
        display: flex; align-items: center; justify-content: center;
    }
    
    .stRadio > div > label:hover {
        background: rgba(255,255,255,0.25); transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: rgba(255,255,255,0.9); color: #2d3748; font-weight: 700;
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
    
    /* Fix for overlapping VaR/CVaR text */
    .risk-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .greeks-alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        font-weight: 600; box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .greeks-alert-warning {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        font-weight: 600; box-shadow: 0 4px 15px rgba(254, 202, 87, 0.3);
    }
    
    .greeks-alert-success {
        background: linear-gradient(135deg, #48cae4 0%, #023047 100%);
        color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        font-weight: 600; box-shadow: 0 4px 15px rgba(72, 202, 228, 0.3);
    }
    
    .greeks-metric-card {
        background: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center;
        transition: all 0.3s ease; border-left: 4px solid var(--accent-color, #667eea);
    }
    
    .greeks-metric-card:hover {
        transform: translateY(-3px); box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    .greeks-value-positive { color: #48bb78; font-weight: 700; }
    .greeks-value-negative { color: #ff6b6b; font-weight: 700; }
    .greeks-value-neutral { color: #667eea; font-weight: 700; }
    
    .risk-gauge-container {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
        padding: 2rem; border-radius: 20px; margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
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
    
    # Centered mode selection with better styling
    col1, col2, col3 = st.columns([1, 1, 1])
    
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
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.portfolio_mode:
        portfolio_interface()
    else:
        single_position_interface()


def create_sample_options_portfolio():
    """Create a sample portfolio with options positions for testing Greeks dashboard."""
    portfolio = PortfolioManager()
    
    # Add some options positions
    portfolio.add_position("oil_hedge", Position(
        commodity="WTI Crude Oil",
        size=1000,  # Long 1000 barrels
        hedge_ratio=0.8,  # 80% hedged
        strategy="Options",
        strike_price=80.0
    ))
    
    portfolio.add_position("gas_protection", Position(
        commodity="Natural Gas",
        size=-500,  # Short 500 MMBtu
        hedge_ratio=0.6,  # 60% hedged
        strategy="Options", 
        strike_price=2.8
    ))
    
    portfolio.add_position("brent_collar", Position(
        commodity="Brent Crude Oil",
        size=800,  # Long 800 barrels
        hedge_ratio=0.9,  # 90% hedged
        strategy="Options",
        strike_price=82.0
    ))
    
    return portfolio

def portfolio_interface():
    """Portfolio management interface."""
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        portfolio_builder_sidebar()
    with col2:
        portfolio_dashboard()


def portfolio_builder_sidebar():
    """Portfolio builder sidebar - FIXED VERSION."""
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
            if 'sample_choice' not in st.session_state:
                st.session_state.sample_choice = "Basic Portfolio"
            
            sample_type = st.selectbox(
                "Sample Type:",
                ["Basic Portfolio", "Options Portfolio"],
                key="sample_type_selector",
                index=0 if st.session_state.get('sample_choice', 'Basic Portfolio') == "Basic Portfolio" else 1
            )
            
            if sample_type == "Basic Portfolio":
                st.session_state.portfolio_manager = create_sample_portfolio()
            else:
                st.session_state.portfolio_manager = create_sample_options_portfolio()
            
            st.session_state.sample_choice = sample_type
            st.success(f"✅ {sample_type} loaded!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            st.session_state.portfolio_manager.clear()
            st.success("✅ Portfolio cleared!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">➕ Add New Position</div>', unsafe_allow_html=True)

    strategy = st.selectbox(
        "Strategy:",
        options=["Futures", "Options"],
        help="Hedging instrument",
        key="strategy_selector"
    )
    
    strike_price = None
    option_expiry = None
    
    if strategy == "Options":
        st.markdown("**⚙️ Options Parameters:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                preview_commodity = "WTI Crude Oil"
                current_price = get_current_price(preview_commodity)
                strike_price = st.slider(
                    "Strike Price ($):",
                    min_value=float(current_price * 0.7),
                    max_value=float(current_price * 1.3),
                    value=float(current_price),
                    step=0.5,
                    help="Option strike price",
                    key="portfolio_strike_price"
                )
                
                moneyness = current_price / strike_price
                if abs(moneyness - 1.0) < 0.05:
                    st.caption("🎯 At-the-Money (ATM)")
                elif moneyness > 1.05:
                    st.caption("📉 Out-of-the-Money (OTM)")
                else:
                    st.caption("📈 In-the-Money (ITM)")
                    
            except Exception as e:
                st.warning(f"Could not fetch current price: {e}")
                strike_price = st.number_input(
                    "Strike Price ($):", 
                    value=75.0, 
                    min_value=1.0, 
                    max_value=200.0,
                    step=0.5,
                    key="portfolio_strike_fallback"
                )
        
        with col2:
            option_expiry = st.selectbox(
                "Option Maturity:",
                options=[1, 3, 6, 12],
                index=1,
                format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
                help="Time until option expiration",
                key="portfolio_option_expiry"
            )
            
            st.caption("📊 Option type will be determined by position direction")
        
        st.markdown("---")
    
    hedge_ratio = st.slider(
        "Hedge Ratio:",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=0.05,
        format="%.2f%%",
        help="Percentage of position to hedge",
        key="hedge_ratio_outside"
    ) 
    st.caption(f"Hedging {hedge_ratio:.2f}% of the position")
    
    with st.form("add_position_form", clear_on_submit=True):
        position_name = st.text_input(
            "Position Name:",
            placeholder="e.g., 'oil_main', 'gas_hedge'",
            help="Unique identifier for this position"
        )
        
        commodity = st.selectbox(
            "Commodity:",
            options=["WTI Crude Oil", "Brent Crude Oil", "Natural Gas"],
            help="Select commodity type",
            key="commodity_selector_form"
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
        
        st.info(f"🛡️ Strategy: {strategy} | Hedge Ratio: {hedge_ratio:.1f}%")
        
        if strategy == "Options" and strike_price:
            option_type = "Put" if position_direction == "Long" else "Call"
            st.info(f"📈 {option_type} Option | Strike: ${strike_price:.2f} | Maturity: {option_expiry} months")
        
        submitted = st.form_submit_button("🔥 Add Position", type="primary", use_container_width=True)
        
        if submitted and position_name:
            if position_name not in st.session_state.portfolio_manager.positions:
                # Adjust strike price based on selected commodity if different from preview
                final_strike_price = strike_price
                if strategy == "Options" and commodity != "WTI Crude Oil":
                    try:
                        commodity_price = get_current_price(commodity)
                        final_strike_price = commodity_price
                    except:
                        final_strike_price = strike_price
                
                new_position = Position(
                    commodity=commodity,
                    size=position_size,
                    hedge_ratio=hedge_ratio/100.0,  # Convert percentage to decimal
                    strategy=strategy,
                    strike_price=final_strike_price
                )
                
                st.session_state.portfolio_manager.add_position(position_name, new_position)
                st.success(f"✅ Added {position_name} to portfolio!")
                st.rerun()
            else:
                st.error(f"❌ Position '{position_name}' already exists!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    """Portfolio dashboard with enhanced Greeks monitoring."""
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔗 Correlations", 
        "⚠️ Risk Analysis", 
        "📈 Greeks Monitor",  # NEW TAB
        "🧪 Stress Testing"
    ])
    
    with tab1:
        portfolio_overview_tab(portfolio, analysis_ready)
    with tab2:
        correlations_tab(portfolio, analysis_ready)
    with tab3:
        risk_analysis_tab(portfolio, analysis_ready)
    with tab4:  # NEW TAB IMPLEMENTATION
        render_enhanced_greeks_tab(portfolio, analysis_ready)
    with tab5:
        stress_testing_tab(portfolio, analysis_ready)


def portfolio_overview_tab(portfolio, analysis_ready):
    """Portfolio overview tab with improved risk metrics display."""
    st.markdown('<div class="section-header">📊 Portfolio Composition</div>', unsafe_allow_html=True)
    
    if analysis_ready:
        risk_summary = portfolio.get_portfolio_risk_summary()
        if risk_summary:
            # Display metrics in a grid to avoid overlapping
            st.markdown('<div class="risk-metrics-grid">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            metrics = list(risk_summary.items())
            
            if len(metrics) > 0:
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #4ECDC4;">Expected P&L</div>
                        <div class="metric-value">{metrics[0][1]}</div>
                        <div class="metric-subtitle">Expected Return</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(metrics) > 1:
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #FF6B6B;">VaR (95%)</div>
                        <div class="metric-value">{metrics[1][1]}</div>
                        <div class="metric-subtitle">Value at Risk</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(metrics) > 2:
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #dc3545;">CVaR (95%)</div>
                        <div class="metric-value">{metrics[2][1]}</div>
                        <div class="metric-subtitle">Tail Risk</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(metrics) > 3:
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #667eea;">Volatility</div>
                        <div class="metric-value">{metrics[3][1]}</div>
                        <div class="metric-subtitle">Risk Level</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Second row of metrics
            if len(metrics) > 4:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title" style="color: #48bb78;">Sharpe Ratio</div>
                        <div class="metric-value">{metrics[4][1]}</div>
                        <div class="metric-subtitle">Risk-Adj. Return</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(metrics) > 5:
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title" style="color: #8e44ad;">Max Drawdown</div>
                            <div class="metric-value">{metrics[5][1]}</div>
                            <div class="metric-subtitle">Worst Loss</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if len(metrics) > 6:
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title" style="color: #2c3e50;">Positions</div>
                            <div class="metric-value">{metrics[6][1]}</div>
                            <div class="metric-subtitle">Total Count</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if len(metrics) > 7:
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title" style="color: #16a085;">Total Notional</div>
                            <div class="metric-value">{metrics[7][1]}</div>
                            <div class="metric-subtitle">Portfolio Size</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
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


def risk_analysis_tab(portfolio, analysis_ready):
    """Risk analysis tab with improved metrics display."""
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
                annotation_position="bottom"
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
            
            st.markdown("### 📊 Comprehensive Risk Metrics")
            
            # First row of risk metrics
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
            
            # Second row of risk metrics
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
                max_loss = np.min(portfolio_pnl)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" style="color: #e74c3c;">Max Loss</div>
                    <div class="metric-value">${max_loss:,.0f}</div>
                    <div class="metric-subtitle">Worst Case</div>
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


def single_position_interface():
    """Improved single position interface."""
    st.markdown('<div class="section-header">🎯 Single Position Analysis</div>', unsafe_allow_html=True)
    
    # Better layout with wider main area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🏭 Position Setup</div>', unsafe_allow_html=True)
        
        commodity = st.selectbox(
            "Select Commodity:",
            options=["WTI Crude Oil", "Brent Crude Oil", "Natural Gas"],
            index=0,
            help="Select the commodity for analysis"
        )
        
        col_dir, col_size = st.columns(2)
        
        with col_dir:
            position_direction = st.selectbox(
                "Direction:",
                options=["Long", "Short"],
                help="Position direction"
            )
        
        with col_size:
            position_size = st.number_input(
                "Size:",
                min_value=1.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0,
                help="Position size"
            )
        
        if position_direction == "Short":
            position_size = -position_size
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🛡️ Hedging Strategy</div>', unsafe_allow_html=True)
        
        strategy = st.selectbox(
            "Strategy:",
            options=["Futures", "Options"],
            help="Hedging instrument"
        )
        
        # Fixed hedge ratio slider with proper formatting
        hedge_ratio = st.slider(
            "Hedge Ratio:",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Percentage of position to hedge"
        )
        
        st.caption(f"Hedging {hedge_ratio*100:.0f}% of the position")
        
        strike_price = None
        option_expiry = None
        
        if strategy == "Options":
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
                
                option_expiry = st.selectbox(
                    "Option Maturity:",
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
        st.markdown('<div class="sidebar-title">📊 Analysis Settings</div>', unsafe_allow_html=True)
        
        confidence = st.slider(
            "Confidence Level:",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            format="%d%%"
        ) / 100.0
        
        n_simulations = st.selectbox(
            "Simulations:",
            options=[1000, 5000, 10000],
            index=1,
            format_func=lambda x: f"{x:,}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state.simulation_run = True
            st.rerun()
    
    with col2:
        if st.session_state.simulation_run:
            with st.spinner("🔄 Running simulation..."):
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
                    
                    st.success("✅ Analysis completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.simulation_run = False
        
        # Display results
        if st.session_state.single_position_results:
            display_single_position_results()
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🎯</div>
                <div class="empty-state-title">Single Position Analysis</div>
                <div class="empty-state-text">
                    Configure your position and click "Run Analysis" to see detailed 
                    payoff diagrams, risk metrics, and Monte Carlo simulations.
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_single_position_results():
    """Display single position analysis results."""
    results = st.session_state.single_position_results
    params = results['params']
    
    # Quick summary
    st.markdown("### 📋 Position Summary")
    
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
    
    # Tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["📊 Payoff Analysis", "📈 Risk Metrics", "🧪 Stress Tests"])
    
    with tab1:
        display_payoff_analysis(results)
    with tab2:
        display_risk_metrics(results)
    with tab3:
        display_stress_tests(results)


def display_payoff_analysis(results):
    """Display payoff analysis charts."""
    payoff_data = results['payoff_data']
    
    # Payoff diagram
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
    fig_payoff.add_vline(x=results['current_price'], line_dash="dash", line_color="orange")
    
    fig_payoff.update_layout(
        title="Payoff Diagram at Expiry",
        xaxis_title="Spot Price ($)",
        yaxis_title="P&L ($)",
        height=400,
        font=dict(family="Inter")
    )
    
    st.plotly_chart(fig_payoff, use_container_width=True)


def display_risk_metrics(results):
    """Display comprehensive risk metrics."""
    hedged_risk = results['hedged_risk']
    unhedged_risk = results['unhedged_risk']
    
    # Risk comparison table
    risk_comparison = summarize_risk_comparison(hedged_risk, unhedged_risk)
    
    st.markdown("### 📊 Risk Metrics Comparison")
    st.dataframe(risk_comparison, use_container_width=True, hide_index=True)
    
    # Hedging effectiveness
    effectiveness = compare_hedging_effectiveness(
        results['sim_results']['hedged_pnl'], 
        results['sim_results']['unhedged_pnl']
    )
    
    st.markdown("### 🎯 Hedging Effectiveness")
    
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


def display_stress_tests(results):
    """Display stress test results for single position."""
    params = results['params']
    current_price = results['current_price']
    
    st.markdown("### 🧪 Historical Crisis Stress Testing")
    
    selected_scenarios = st.multiselect(
        "Select crisis scenarios:",
        options=list(STRESS_SCENARIOS.keys()),
        default=list(STRESS_SCENARIOS.keys())[:3],
        key="single_stress"
    )
    
    if selected_scenarios:
        stress_results = []
        
        for scenario_name in selected_scenarios:
            scenario = STRESS_SCENARIOS[scenario_name]
            price_change = scenario["oil_peak_to_trough"]
            
            position_size = abs(params['position'])
            price_shock = current_price * price_change
            
            unhedged_pnl = price_shock * position_size
            
            hedge_ratio = params['hedge_ratio']
            if params['strategy'] == "Futures":
                hedge_pnl = -price_shock * position_size * hedge_ratio
            elif params['strategy'] == "Options":
                if price_change < 0:
                    hedge_pnl = -price_shock * position_size * hedge_ratio * 0.8
                else:
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
        
        st.dataframe(pd.DataFrame(stress_results), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
