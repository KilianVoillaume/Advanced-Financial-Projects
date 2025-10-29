"""
app.py - Enhanced Oil & Gas Hedging Platform with Multi-Leg Options Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from hedging.data import get_prices, get_current_price, get_available_commodities
from hedging.strategies import compute_payoff_diagram, get_hedge_summary, compute_multi_leg_payoff
from hedging.simulation import simulate_hedged_vs_unhedged, compare_hedging_effectiveness
from hedging.risk import calculate_risk_metrics, calculate_delta_exposure, summarize_risk_comparison
from hedging.stress_testing import STRESS_SCENARIOS
from hedging.portfolio import (PortfolioManager, Position, create_oil_position, create_gas_position, 
                              create_brent_position, create_sample_portfolio)
from hedging.greeks_dashboard import GreeksDashboard, GreeksMonitor, render_enhanced_greeks_tab
from hedging.multi_leg_strategies import (StrategyType, create_long_straddle, create_short_straddle,
                                         create_long_strangle, create_short_strangle, create_collar,
                                         create_butterfly_spread, create_iron_condor,
                                         create_strategy_from_preset, get_strategy_defaults)
from hedging.multi_leg_strategies import validate_multi_leg_configuration
from hedging.ui_components import render_enhanced_commodity_selector, render_commodity_info_panel
from hedging.pnl_attribution import render_pnl_attribution_tab, PnLAttributionEngine, PnLAttributionUI


st.set_page_config(
    page_title="Commodity Hedging Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with multi-leg strategy support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* PREMIUM COLOR PALETTE - Financial Industry Standard */
    :root {
        /* Primary Colors - Professional Blues */
        --primary-50: #f0f6ff;
        --primary-100: #e0edff;
        --primary-500: #3b82f6;
        --primary-600: #2563eb;
        --primary-700: #1d4ed8;
        --primary-900: #1e3a8a;
        
        /* Neutral Grays - Sophisticated */
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-500: #6b7280;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
        
        /* Success/Danger - Financial */
        --success-500: #10b981;
        --success-600: #059669;
        --danger-500: #ef4444;
        --danger-600: #dc2626;
        --warning-500: #f59e0b;
        --warning-600: #d97706;
        
        /* Shadows - Depth */
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }

    /* CRITICAL FIX: Prevent overlapping and transparency issues */
    .main > div {
        position: relative !important;
        z-index: 1 !important;
        background: var(--gray-50) !important;
    }
    
    /* Fix for Streamlit container stacking */
    .stContainer > div {
        position: relative !important;
        z-index: auto !important;
    }
    
    /* Ensure proper tab content isolation */
    .stTabs > div > div > div > div {
        background: transparent !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    /* Clear any duplicate background elements */
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
        position: relative !important;
        z-index: 3 !important;
        clear: both !important;
    }

    .main-container { 
        font-family: 'Inter', sans-serif; 
        background: var(--gray-50) !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* HERO HEADER - Premium Financial Look */
    .hero-header-premium {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
        z-index: 10 !important;
    }

    .hero-header-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-width="0.5" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)" /></svg>');
        pointer-events: none;
        z-index: -1;
    }

    .hero-title-premium {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.025em;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle-premium {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        max-width: 800px;
        margin: 0 auto;
    }

    /* PREMIUM CARDS - Institutional Grade */
    .premium-card {
        background: white !important;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--gray-200);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative !important;
        overflow: hidden;
        margin-bottom: 1.5rem;
        z-index: 5 !important;
    }

    .premium-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
        border-color: var(--primary-300);
    }

    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-500), var(--primary-600));
        z-index: -1;
    }

    /* PORTFOLIO OVERVIEW CARD - Executive Style */
    .portfolio-overview-premium {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%) !important;
        border-radius: 20px;
        padding: 3rem;
        color: white;
        box-shadow: var(--shadow-xl);
        position: relative !important;
        overflow: hidden;
        margin-bottom: 2rem;
        z-index: 5 !important;
    }

    .portfolio-overview-premium::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
        z-index: -1;
    }

    .portfolio-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
        position: relative;
        z-index: 1;
    }

    .portfolio-stat {
        text-align: center;
        position: relative;
        z-index: 1;
    }

    .portfolio-stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }

    .portfolio-stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }

    /* METRIC CARDS - Professional */
    .metric-card-premium {
        background: white !important;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--gray-200);
        transition: all 0.3s ease;
        text-align: center;
        position: relative !important;
        z-index: 5 !important;
    }

    .metric-card-premium:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .metric-title-premium {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }

    .metric-value-premium {
        font-size: 2rem;
        font-weight: 700;
        color: var(--gray-900);
        margin-bottom: 0.25rem;
        line-height: 1;
    }

    .metric-subtitle-premium {
        font-size: 0.75rem;
        color: var(--gray-500);
        opacity: 0.8;
    }

    /* INFO BOXES */
    .info-box {
        background: linear-gradient(135deg, var(--primary-50), var(--primary-100));
        border-left: 4px solid var(--primary-500);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: var(--primary-900);
        font-weight: 500;
        position: relative;
        z-index: 5;
    }

    /* ALERT STYLES */
    .alert-premium {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        margin: 1rem 0;
        font-weight: 500;
        position: relative;
        z-index: 5;
    }

    .alert-success-premium {
        background: rgba(16, 185, 129, 0.05);
        border-left-color: var(--success-500);
        color: var(--success-700);
    }

    .alert-warning-premium {
        background: rgba(245, 158, 11, 0.05);
        border-left-color: var(--warning-500);
        color: var(--warning-700);
    }

    .alert-danger-premium {
        background: rgba(239, 68, 68, 0.05);
        border-left-color: var(--danger-500);
        color: var(--danger-700);
    }

    /* SIDEBAR IMPROVEMENTS */
    .sidebar-section-premium {
        background: white !important;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--gray-200);
        position: relative !important;
        z-index: 5 !important;
    }

    .sidebar-title-premium {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--gray-900);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* POSITION CARDS */
    .position-card-premium {
        background: white !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--gray-200);
        transition: all 0.2s ease;
        position: relative !important;
        z-index: 5 !important;
    }

    .position-card-premium:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }

    .position-card-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(to bottom, var(--primary-500), var(--primary-600));
        border-radius: 12px 0 0 12px;
        z-index: -1;
    }

    /* ANIMATIONS - Smooth Professional */
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* LOADING STATES */
    .loading-container-premium {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 3rem;
        background: white !important;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        position: relative;
        z-index: 5;
    }

    .loading-spinner-premium {
        display: inline-block;
        width: 24px;
        height: 24px;
        border: 3px solid var(--gray-300);
        border-radius: 50%;
        border-top-color: var(--primary-600);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* RESPONSIVE DESIGN */
    @media (max-width: 1024px) {
        .hero-title-premium { font-size: 2.5rem; }
        .portfolio-stats-grid { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 768px) {
        .hero-title-premium { font-size: 2rem; }
        .portfolio-stats-grid { grid-template-columns: 1fr; }
        .premium-card { padding: 1.5rem; }
    }

    /* STREAMLIT OVERRIDES - CRITICAL FIXES */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 1.75rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-md) !important;
        position: relative !important;
        z-index: 10 !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-lg) !important;
        background: linear-gradient(135deg, var(--primary-700), var(--primary-800)) !important;
    }

    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid var(--gray-300) !important;
        background: white !important;
        position: relative !important;
        z-index: 5 !important;
    }

    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid var(--gray-300) !important;
        padding: 0.875rem 1rem !important;
        background: white !important;
        position: relative !important;
        z-index: 5 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* TAB STYLING - CRITICAL FIX */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: white !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        box-shadow: var(--shadow-sm) !important;
        position: relative !important;
        z-index: 100 !important;
        margin-bottom: 1rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem !important;
        padding: 0 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        background: transparent !important;
        color: var(--gray-600) !important;
        position: relative !important;
        z-index: 101 !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-600) !important;
        color: white !important;
        z-index: 102 !important;
    }
    
    /* CRITICAL: Ensure tab content doesn't overlap */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem !important;
        background: transparent !important;
        position: relative !important;
        z-index: 50 !important;
    }
    
    /* Fix any phantom overlays */
    .stTabs [data-baseweb="tab-panel"] > div {
        background: transparent !important;
        position: relative !important;
        z-index: 51 !important;
    }

    /* Ensure main content area is properly isolated */
    .block-container {
        background: var(--gray-50) !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* Clear any floating elements */
    .stTabs::after {
        content: "" !important;
        display: table !important;
        clear: both !important;
    }
</style>
""", unsafe_allow_html=True)


class SessionStateManager:
    """Manages Streamlit session state for the hedging platform"""
    
    @staticmethod
    def should_recalculate_greeks() -> bool:
        """
        Determine if Greeks should be recalculated based on portfolio changes
        """
        # Check if this is the first calculation
        if 'greeks_calculated' not in st.session_state:
            return True
        
        # Check if portfolio has changed since last calculation
        current_portfolio_hash = SessionStateManager._get_portfolio_hash()
        last_portfolio_hash = st.session_state.get('last_portfolio_hash', '')
        
        if current_portfolio_hash != last_portfolio_hash:
            return True
        
        # Check if calculation was done recently (avoid recalculating too frequently)
        last_calc_time = st.session_state.get('last_greeks_calc_time', 0)
        current_time = datetime.now().timestamp()
        
        # Recalculate if more than 30 seconds have passed
        if current_time - last_calc_time > 30:
            return True
        
        return False
    
    @staticmethod
    def mark_greeks_calculated():
        """
        Mark that Greeks have been calculated and store the current state
        """
        st.session_state.greeks_calculated = True
        st.session_state.last_greeks_calc_time = datetime.now().timestamp()
        st.session_state.last_portfolio_hash = SessionStateManager._get_portfolio_hash()
    
    @staticmethod
    def _get_portfolio_hash() -> str:
        """
        Generate a hash of the current portfolio state to detect changes
        """
        try:
            portfolio_manager = st.session_state.get('portfolio_manager')
            if not portfolio_manager or len(portfolio_manager.positions) == 0:
                return "empty_portfolio"
            
            # Create a simple hash based on position data
            position_data = []
            for name, position in portfolio_manager.positions.items():
                pos_info = f"{name}:{position.commodity}:{position.size}:{position.hedge_ratio}:{position.strategy}"
                if position.multi_leg_strategy:
                    pos_info += f":{position.multi_leg_strategy.strategy_type.value}"
                position_data.append(pos_info)
            
            portfolio_string = "|".join(sorted(position_data))
            return str(hash(portfolio_string))
        
        except Exception as e:
            # Fallback to timestamp if hashing fails
            return str(datetime.now().timestamp())
    
    @staticmethod
    def reset_calculations():
        """
        Reset all calculation flags - useful when portfolio is cleared
        """
        keys_to_remove = [
            'greeks_calculated',
            'last_greeks_calc_time', 
            'last_portfolio_hash',
            'analysis_ready'
        ]
        
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def force_recalculation():
        """
        Force the next calculation cycle by clearing the calculated flag
        """
        if 'greeks_calculated' in st.session_state:
            del st.session_state['greeks_calculated']

def initialize_session_state():
    if 'portfolio_manager' not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()


def render_butterfly_parameters(current_price, defaults):
    """Render butterfly spread-specific parameters."""
    col1, col2, col3 = st.columns(3)
    
    strikes = defaults.get('strikes', {})
    
    with col1:
        lower_strike = st.slider(
            "Lower Strike ($):",
            min_value=float(current_price * 0.8),
            max_value=float(current_price * 0.98),
            value=float(strikes.get('lower_strike', current_price * 0.95)),
            step=0.5,
            key="butterfly_lower"
        )
    
    with col2:
        middle_strike = st.slider(
            "Middle Strike ($):",
            min_value=float(current_price * 0.98),
            max_value=float(current_price * 1.02),
            value=float(strikes.get('middle_strike', current_price)),
            step=0.5,
            key="butterfly_middle"
        )
    
    with col3:
        upper_strike = st.slider(
            "Upper Strike ($):",
            min_value=float(current_price * 1.02),
            max_value=float(current_price * 1.2),
            value=float(strikes.get('upper_strike', current_price * 1.05)),
            step=0.5,
            key="butterfly_upper"
        )
    
    # Validate strike ordering
    if not (lower_strike < middle_strike < upper_strike):
        st.warning("⚠️ Strikes must be ordered: Lower < Middle < Upper")
    
    width = min(middle_strike - lower_strike, upper_strike - middle_strike)
    st.caption(f"💰 Wing width: ${width:.2f}")
    
    st.session_state.multi_leg_config = {
        'strategy_type': 'Butterfly Spread',
        'lower_strike': lower_strike,
        'middle_strike': middle_strike,
        'upper_strike': upper_strike,
        'expiry_months': 3
    }


def render_iron_condor_parameters(current_price, defaults):
    """Render iron condor-specific parameters."""
    strikes = defaults.get('strikes', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Put Spread:**")
        put_strike_low = st.slider(
            "Put Strike Low ($):",
            min_value=float(current_price * 0.7),
            max_value=float(current_price * 0.9),
            value=float(strikes.get('put_strike_low', current_price * 0.85)),
            step=0.5,
            key="condor_put_low"
        )
        
        put_strike_high = st.slider(
            "Put Strike High ($):",
            min_value=float(current_price * 0.85),
            max_value=float(current_price * 0.98),
            value=float(strikes.get('put_strike_high', current_price * 0.95)),
            step=0.5,
            key="condor_put_high"
        )
    
    with col2:
        st.markdown("**Call Spread:**")
        call_strike_low = st.slider(
            "Call Strike Low ($):",
            min_value=float(current_price * 1.02),
            max_value=float(current_price * 1.15),
            value=float(strikes.get('call_strike_low', current_price * 1.05)),
            step=0.5,
            key="condor_call_low"
        )
        
        call_strike_high = st.slider(
            "Call Strike High ($):",
            min_value=float(current_price * 1.1),
            max_value=float(current_price * 1.3),
            value=float(strikes.get('call_strike_high', current_price * 1.15)),
            step=0.5,
            key="condor_call_high"
        )
    
    # Validate strike ordering
    if not (put_strike_low < put_strike_high < call_strike_low < call_strike_high):
        st.warning("⚠️ Invalid strike configuration")
    else:
        profit_zone = call_strike_low - put_strike_high
        st.caption(f"💰 Profit zone width: ${profit_zone:.2f}")
    
    st.session_state.multi_leg_config = {
        'strategy_type': 'Iron Condor',
        'put_strike_low': put_strike_low,
        'put_strike_high': put_strike_high,
        'call_strike_low': call_strike_low,
        'call_strike_high': call_strike_high,
        'expiry_months': 3
    }


def render_single_option_parameters():
    """Render single option parameters."""
    st.markdown("**⚙️ Options Parameters:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            current_price = get_current_price("WTI Crude Oil")
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
        option_type = st.selectbox(
            "Option Type:",
            options=["Put", "Call"],
            index=0,
            help="Type of option contract",
            key="portfolio_option_type"
        )
        
        option_expiry = st.selectbox(
            "Option Maturity:",
            options=[1, 3, 6, 12],
            index=1,
            format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
            help="Time until option expiration",
            key="portfolio_option_expiry"
        )


def create_position_from_form(position_name, commodity, position_size, hedge_ratio, strategy):
    """Create a Position object from form inputs."""
    if strategy == "Multi-Leg Options":
        config = st.session_state.get('multi_leg_config', {})
        if not config:
            raise ValueError("Multi-leg strategy configuration missing")
        
        # Get current price and convert position size to contract units
        strategy_type = config['strategy_type']
        try:
            current_price = get_current_price(commodity)
        except:
            # Fallback prices if data unavailable
            fallback_prices = {
                "WTI Crude Oil": 75.0,
                "Brent Crude Oil": 78.0,
                "Natural Gas": 3.5,
                "Copper": 4.0,
                "Gold": 2000.0,
                "Silver": 25.0
            }
            current_price = fallback_prices.get(commodity, 75.0)
        
        # CRITICAL FIX: Convert dollar position to contract units
        contracts = abs(position_size) / current_price
        
        # Create multi-leg strategy based on type with correct position sizing
        if strategy_type == "Long Straddle":
            multi_leg_strategy = create_long_straddle(
                config['strike_price'], contracts, hedge_ratio, commodity
            )
        elif strategy_type == "Short Straddle":
            multi_leg_strategy = create_short_straddle(
                config['strike_price'], contracts, hedge_ratio, commodity
            )
        elif strategy_type == "Long Strangle":
            multi_leg_strategy = create_long_strangle(
                config['call_strike'], config['put_strike'], 
                contracts, hedge_ratio, commodity
            )
        elif strategy_type == "Short Strangle":
            multi_leg_strategy = create_short_strangle(
                config['call_strike'], config['put_strike'], 
                contracts, hedge_ratio, commodity
            )
        elif strategy_type == "Collar":
            multi_leg_strategy = create_collar(
                config['call_strike'], config['put_strike'],
                contracts, hedge_ratio, commodity
            )
        elif strategy_type == "Butterfly Spread":
            multi_leg_strategy = create_butterfly_spread(
                config['lower_strike'], config['middle_strike'], config['upper_strike'],
                contracts, hedge_ratio, commodity
            )
        elif strategy_type == "Iron Condor":
            multi_leg_strategy = create_iron_condor(
                config['put_strike_low'], config['put_strike_high'],
                config['call_strike_low'], config['call_strike_high'],
                contracts, hedge_ratio, commodity
            )
        else:
            raise ValueError(f"Unknown multi-leg strategy: {strategy_type}")
        
        # Create position with multi-leg strategy
        new_position = Position(
            commodity=commodity,
            size=position_size,  # Keep original dollar size for notional calculations
            hedge_ratio=hedge_ratio,
            strategy="Multi-Leg",
            multi_leg_strategy=multi_leg_strategy
        )
    
    elif strategy == "Options":
        strike_price = st.session_state.get('portfolio_strike_price', 75.0)
        option_type = st.session_state.get('portfolio_option_type', 'Put')
        
        return Position(
            commodity=commodity,
            size=position_size,
            hedge_ratio=hedge_ratio,
            strategy="Options",
            strike_price=strike_price,
            option_type=option_type
        )
    
    else:  # Futures
        return Position(
            commodity=commodity,
            size=position_size,
            hedge_ratio=hedge_ratio,
            strategy="Futures"
        )


def estimate_strategy_cost(strategy_type, current_price, strike_price):
    """Estimate the cost of a strategy (simplified)."""
    # This is a simplified estimation - in production would use Black-Scholes
    time_value = current_price * 0.02  # 2% time value
    
    if "Straddle" in strategy_type:
        # Call + Put at same strike
        call_intrinsic = max(current_price - strike_price, 0)
        put_intrinsic = max(strike_price - current_price, 0)
        return call_intrinsic + put_intrinsic + (2 * time_value)
    
    elif "Strangle" in strategy_type:
        # Simplified - assume OTM options
        return 2 * time_value
    
    elif "Collar" in strategy_type:
        # Put premium minus call premium (simplified)
        return time_value * 0.5
    
    else:
        return time_value


def render_current_positions_list(portfolio):
    """Render current positions list with enhanced multi-leg display."""
    st.markdown("### 📋 Current Positions")
    
    if len(portfolio.positions) == 0:
        st.info("No positions added yet. Use the form above to add positions.")
        return

    for name, position in portfolio.positions.items():
        direction_emoji = "📈" if position.size > 0 else "📉"
        direction_text = "Long" if position.size > 0 else "Short"
        hedge_status = "🛡️" if position.hedge_ratio > 0.5 else "⚠️" if position.hedge_ratio > 0 else "🚫"
        
        # Enhanced strategy description
        if position.is_multi_leg:
            strategy_desc = position.multi_leg_strategy.strategy_type.value
            legs_count = len(position.multi_leg_strategy.legs)
            strategy_info = f"{strategy_desc} ({legs_count} legs)"
        else:
            strategy_info = position.strategy
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            if position.hedge_ratio >= 0.8:
                hedge_status = "🛡️ Well Hedged"
                hedge_color = "var(--success-600)"
            elif position.hedge_ratio >= 0.5:
                hedge_status = "⚠️ Partially Hedged" 
                hedge_color = "var(--warning-600)"
            else:
                hedge_status = "🚫 Low Hedge"
                hedge_color = "var(--danger-600)"

            st.markdown(f"""
            <div class="position-card-premium">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.3rem;">
                    {direction_emoji} {name}
                </div>
                <div style="color: #666; font-size: 0.9rem; line-height: 1.4;">
                    {position.commodity}<br>
                    {direction_text} {abs(position.size):,.0f} • {strategy_info}<br>
                    <span style="color: {hedge_color}; font-weight: 600;">{position.hedge_ratio*100:.0f}% {hedge_status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("❌", key=f"remove_{name}", help=f"Remove {name}"):
                st.session_state.portfolio_manager.remove_position(name)
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_portfolio_analysis_settings():
    """Render portfolio analysis settings."""
    st.markdown("### ⚙️ Analysis Settings")

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
    
    time_horizon = st.selectbox(
        "Risk Measurement Period:",
        options=["1-Day", "1-Week", "1-Month"],
        index=0,
        help="Time horizon for risk calculations"
    )

    st.session_state.portfolio_manager.set_config(
        confidence_level=confidence_level,
        simulation_runs=n_simulations
    )
    
    st.info(f"""
    📊 **Current Settings:**
    - Risk Horizon: **{time_horizon}**
    - Simulations: **{n_simulations:,}**
    - Confidence: **{confidence_level:.0%}**
    """)

    st.markdown('</div>', unsafe_allow_html=True)


def portfolio_dashboard():
    """Enhanced portfolio dashboard with overlap prevention"""
    
    # Clear any potential caching issues
    clear_streamlit_cache()
    
    portfolio = st.session_state.portfolio_manager
    
    if len(portfolio) == 0:
        render_enhanced_portfolio_summary(portfolio)
        return

    should_recalculate = SessionStateManager.should_recalculate_greeks()
    
    if should_recalculate:
        with st.spinner("🔄 Calculating portfolio analytics..."):
            try:
                portfolio.calculate_correlations(force_recalculate=True)
                portfolio.calculate_portfolio_risk(force_recalculate=True)
                st.session_state.analysis_ready = True
                SessionStateManager.mark_greeks_calculated()
                st.success("✅ Analytics updated!")
            except Exception as e:
                st.error(f"❌ Error in analytics: {e}")
                st.session_state.analysis_ready = False
    else:
        st.session_state.analysis_ready = True

    render_enhanced_portfolio_summary(portfolio)
    
    tab_container_key = f"portfolio_tabs_{hash(str(portfolio.positions))}"
    
    st.empty()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "💰 P&L Attribution", 
        "🔗 Correlations", 
        "⚠️ Risk Analysis", 
        "📈 Greeks Monitor",
        "🧪 Stress Testing"
    ])
    
    analysis_ready = st.session_state.get('analysis_ready', False)
    
    with tab1:
        st.empty()  
        portfolio_overview_tab(portfolio, analysis_ready)
        
    with tab2:
        st.empty()  
        render_pnl_attribution_tab_with_live_tracker(portfolio)
        
    with tab3:
        st.empty()  
        correlations_tab(portfolio, analysis_ready)
        
    with tab4:
        st.empty()  
        risk_analysis_tab(portfolio, analysis_ready)
        
    with tab5:
        st.empty()  
        render_enhanced_greeks_tab(portfolio, analysis_ready)
        
    with tab6:
        st.empty()  
        stress_testing_tab(portfolio, analysis_ready)


def render_commodity_breakdown_in_summary(portfolio):
    """Add commodity breakdown as an expandable section"""
    
    if len(portfolio.positions) == 0:
        return
        
    # Calculate by commodity
    commodity_data = {}
    for position in portfolio.positions.values():
        commodity = position.commodity
        if commodity not in commodity_data:
            commodity_data[commodity] = {'long': 0, 'short': 0, 'net': 0}
        
        notional = position.notional_value
        if position.size > 0:
            commodity_data[commodity]['long'] += notional
        else:
            commodity_data[commodity]['short'] += abs(notional)
        commodity_data[commodity]['net'] = commodity_data[commodity]['long'] - commodity_data[commodity]['short']
    
    # Display commodity breakdown
    with st.expander("📊 Commodity Breakdown", expanded=False):
        for commodity, data in commodity_data.items():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Commodity", commodity)
            with col2:
                st.metric("Long", f"${data['long']:,.0f}")
            with col3:
                st.metric("Short", f"${data['short']:,.0f}")
            with col4:
                st.metric("Net", f"${data['net']:+,.0f}")


def portfolio_overview_tab(portfolio, analysis_ready):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    st.markdown("### 📊 Portfolio Composition & Daily Risk Metrics")  
    
    if analysis_ready:
        # Force calculation to ensure we have the latest risk metrics
        portfolio.calculate_portfolio_risk(force_recalculate=False)  # Use cache if available
        risk_summary = portfolio.get_portfolio_risk_summary()
        
        if risk_summary:
            st.markdown('<div class="risk-metrics-grid">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            expected_pnl_raw = risk_summary.get('Expected P&L', '$0')
            var_raw = risk_summary.get('VaR (95%)', '$0') 
            cvar_raw = risk_summary.get('CVaR (95%)', '$0')
            volatility_raw = risk_summary.get('Volatility', '$0')
            
            def parse_currency(value_str):
                """Helper to parse currency strings like '$-123' or '$1,234'"""
                try:
                    return float(value_str.replace('$', '').replace(',', ''))
                except:
                    return 0.0
            
            expected_pnl = parse_currency(expected_pnl_raw)
            var_95 = parse_currency(var_raw)
            cvar_95 = parse_currency(cvar_raw)
            volatility = parse_currency(volatility_raw)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #4ECDC4;">Daily Expected P&L</div>
                    <div class="metric-value-premium">${expected_pnl:,.0f}</div>
                    <div class="metric-subtitle-premium">1-Day Average Return</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #FF6B6B;">Daily VaR (95%)</div>
                    <div class="metric-value-premium">${var_95:,.0f}</div>
                    <div class="metric-subtitle-premium">1-Day Value at Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #dc3545;">Daily CVaR (95%)</div>
                    <div class="metric-value-premium">${cvar_95:,.0f}</div>
                    <div class="metric-subtitle-premium">1-Day Tail Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #667eea;">Daily Volatility</div>
                    <div class="metric-value-premium">${volatility:,.0f}</div>
                    <div class="metric-subtitle-premium">1-Day Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### 📅 Extended Time Horizon Estimates")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate extended time horizons
            monthly_expected = expected_pnl * 21  # 21 trading days per month
            annual_expected = expected_pnl * 252  # 252 trading days per year
            
            monthly_var = var_95 * np.sqrt(21)
            annual_var = var_95 * np.sqrt(252)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #48bb78;">Weekly Expected P&L</div>
                    <div class="metric-value-premium">${expected_pnl * 5:,.0f}</div>
                    <div class="metric-subtitle-premium">5-Day Estimate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #48bb78;">Monthly Expected P&L</div>
                    <div class="metric-value-premium">${monthly_expected:,.0f}</div>
                    <div class="metric-subtitle-premium">21-Day Estimate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #48bb78;">Annual Expected P&L</div>
                    <div class="metric-value-premium">${annual_expected:,.0f}</div>
                    <div class="metric-subtitle-premium">252-Day Estimate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                sharpe_raw = risk_summary.get('Sharpe Ratio', '0.000')
                try:
                    sharpe_ratio = float(sharpe_raw)
                except:
                    sharpe_ratio = expected_pnl / volatility if volatility > 0 else 0
                
                st.markdown(f"""
                <div class="metric-card-premium">
                    <div class="metric-title-premium" style="color: #8e44ad;">Daily Sharpe Ratio</div>
                    <div class="metric-value-premium">{sharpe_ratio:.3f}</div>
                    <div class="metric-subtitle-premium">Risk-Adj. Daily Return</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced portfolio composition visualization
    render_enhanced_portfolio_composition(portfolio)


def render_enhanced_portfolio_composition(portfolio):
    """
    Enhanced portfolio composition with consistent data source and improved visualizations.
    """
    if len(portfolio.positions) == 0:
        st.info("📊 Add positions to see portfolio composition")
        return
    
    # Calculate position data
    total_notional = sum(pos.notional_value for pos in portfolio.positions.values())
    
    if total_notional == 0:
        st.warning("⚠️ Portfolio has zero notional value")
        return
    
    # Prepare data for visualizations
    position_data = []
    category_data = {}
    
    for name, position in portfolio.positions.items():
        weight = position.notional_value / total_notional
        
        # Determine category
        if position.is_multi_leg:
            category = "Multi-Leg"
        elif position.strategy == "Options":
            category = "Options"
        else:
            category = "Futures"
        
        position_data.append({
            'Position': name,
            'Weight': weight,
            'Notional': position.notional_value,
            'Category': category,
            'Commodity': position.commodity,
            'Strategy': position.strategy
        })
        
        # Aggregate by category
        if category not in category_data:
            category_data[category] = 0
        category_data[category] += weight
    
    # Create enhanced visualization
    create_portfolio_composition_charts(position_data, category_data)
    render_enhanced_commodity_exposure_display(portfolio)


def create_portfolio_composition_charts(position_data, category_data):
    """
    Create comprehensive portfolio composition charts
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}],
               [{"type": "bar", "colspan": 2}, None]],
        subplot_titles=(
            "Portfolio by Position", 
            "Portfolio by Category",
            "Position Details"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Portfolio by Position (Top Left)
    position_names = [item['Position'] for item in position_data]
    position_weights = [item['Weight'] for item in position_data]
    position_colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#ffecd2', '#fcb69f']
    
    fig.add_trace(
        go.Pie(
            labels=position_names,
            values=position_weights,
            name="By Position",
            marker=dict(colors=position_colors[:len(position_names)]),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Notional: $%{customdata:,.0f}<extra></extra>',
            customdata=[item['Notional'] for item in position_data]
        ),
        row=1, col=1
    )
    
    # 2. Portfolio by Category (Top Right) 
    category_names = list(category_data.keys())
    category_weights = list(category_data.values())
    category_colors = ['#FF6B6B', '#4ECDC4', '#48bb78', '#667eea', '#f39c12', '#9c27b0', '#e74c3c']
    
    fig.add_trace(
        go.Pie(
            labels=category_names,
            values=category_weights,
            name="By Category",
            marker=dict(colors=category_colors[:len(category_names)]),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Position Details Bar Chart (Bottom)
    fig.add_trace(
        go.Bar(
            x=[item['Position'] for item in position_data],
            y=[item['Weight'] for item in position_data],
            name="Position Weights",
            marker_color=[position_colors[i % len(position_colors)] for i in range(len(position_data))],
            text=[f"{w:.1%}" for w in position_weights],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Weight: %{y:.1%}<br>Notional: $%{customdata:,.0f}<br>Category: %{text}<extra></extra>',
            customdata=[item['Notional'] for item in position_data],
            texttemplate=[item['Category'] for item in position_data]
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Enhanced Portfolio Composition Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter', 'color': '#2d3748'}
        },
        height=800,
        showlegend=False,
        font=dict(family="Inter", size=10),
        margin=dict(t=80, b=80, l=60, r=60),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update bar chart layout
    fig.update_xaxes(title_text="Positions", row=2, col=1)
    fig.update_yaxes(title_text="Weight (%)", row=2, col=1, tickformat='.1%')
    
    st.plotly_chart(fig, use_container_width=True)


def render_pnl_attribution_tab_with_live_tracker(portfolio_manager: PortfolioManager):
    """Enhanced P&L Attribution tab with integrated Live P&L Tracker"""
    
    if len(portfolio_manager.positions) == 0:
        st.info("📊 Add positions to see P&L attribution analysis")
        return
    
    st.markdown("### 📊 Live P&L Tracker")
    
    attribution_engine = PnLAttributionEngine(portfolio_manager)
    
    try:
        attribution_results = attribution_engine.calculate_daily_pnl_attribution()
        portfolio_pnl = attribution_results['portfolio_total']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_pnl = portfolio_pnl.get('total_pnl', 0)
            pnl_change = "+$" if total_pnl >= 0 else "-$"
            st.metric(
                "Live P&L", 
                f"${total_pnl:,.0f}",
                f"{pnl_change}{abs(total_pnl):,.0f} today"
            )
        
        with col2:
            delta_pnl = portfolio_pnl.get('delta_pnl', 0)
            st.metric(
                "Delta P&L",
                f"${delta_pnl:,.0f}",
                "Price moves"
            )
        
        with col3:
            theta_pnl = portfolio_pnl.get('theta_pnl', 0)
            st.metric(
                "Theta P&L",
                f"${theta_pnl:,.0f}",
                "Time decay"
            )
        
        render_pnl_alerts(attribution_engine)
        
        st.markdown("---")
        
        PnLAttributionUI.render_pnl_attribution_dashboard(attribution_engine)
        
    except Exception as e:
        st.error(f"Error in live P&L tracking: {e}")

def render_enhanced_commodity_exposure_display(portfolio):
    """Render enhanced commodity exposure with category grouping"""
    
    st.markdown("### 🎯 Commodity Exposure")
    
    exposure_df = portfolio.get_commodity_exposure()
    
    if exposure_df.empty:
        st.info("📊 Add positions to see commodity exposure")
        return
    
    # Display the enhanced exposure table
    st.dataframe(
        exposure_df, 
        use_container_width=True, 
        hide_index=True,
        height=min(400, len(exposure_df) * 40 + 100)
    )
    
    # Add exposure summary
    if not exposure_df.empty:
        render_exposure_summary_metrics(portfolio)


def render_exposure_summary_metrics(portfolio):
    """Render summary metrics for commodity exposure"""
    from hedging.data import get_commodity_category
    
    # Calculate category-level statistics
    category_exposure = {}
    total_long = 0
    total_short = 0
    
    for position in portfolio.positions.values():
        category = get_commodity_category(position.commodity)
        exposure = position.size * position.current_price
        
        if category not in category_exposure:
            category_exposure[category] = 0
        category_exposure[category] += exposure
        
        if exposure > 0:
            total_long += exposure
        else:
            total_short += abs(exposure)
    
    # Display summary
    st.markdown("**📊 Exposure Summary:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Long", f"${total_long:,.0f}")
    
    with col2:
        st.metric("Total Short", f"${total_short:,.0f}")
    
    with col3:
        net_exposure = total_long - total_short
        st.metric("Net Exposure", f"${net_exposure:+,.0f}")
    
    # Category breakdown
    if len(category_exposure) > 1:
        st.markdown("**🏷️ By Category:**")
        for category, exposure in sorted(category_exposure.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "📈" if exposure > 0 else "📉" if exposure < 0 else "⚖️"
            st.write(f"{direction} **{category}**: ${exposure:+,.0f}")

def correlations_tab(portfolio, analysis_ready):
    """Fixed correlations analysis tab"""
    st.markdown('<div class="section-header">🔗 Cross-Commodity Correlations</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to calculate correlations")
        return
    
    # Get unique commodities from portfolio
    unique_commodities = list(set(pos.commodity for pos in portfolio.positions.values()))
    
    if len(unique_commodities) < 2:
        st.info(f"""
        ℹ️ **Need multiple commodities for correlation analysis**
        
        Current commodities: {', '.join(unique_commodities)}
        
        **To see correlations:**
        - Add positions in different commodities (WTI, Brent, Natural Gas)
        - Each commodity needs sufficient price history
        """)
        return
    
    # Try to get correlation matrix
    try:
        corr_matrix = portfolio.get_correlation_matrix()
        
        if corr_matrix.empty or len(corr_matrix) < 2:
            st.warning(f"""
            ⚠️ **Cannot calculate correlations**
            
            **Possible reasons:**
            - Insufficient price data for: {', '.join(unique_commodities)}
            - Network issues fetching commodity prices
            - All positions in same commodity
            
            **Solutions:**
            - Add positions in different commodities
            - Check internet connection
            - Try refreshing the page
            """)
            return
        
        # Display correlation heatmap
        import plotly.graph_objects as go
        
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
                'text': f"Commodity Correlation Matrix ({len(unique_commodities)} commodities)",
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
        
        # Display correlation summary
        st.markdown("### 📊 Correlation Summary")
        
        correlation_insights = []
        for i, commodity1 in enumerate(corr_matrix.columns):
            for j, commodity2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strength = "Strong"
                        color = "🔴" if corr_value > 0 else "🔵"
                    elif abs(corr_value) > 0.3:
                        strength = "Moderate"
                        color = "🟡"
                    else:
                        strength = "Weak"
                        color = "🟢"
                    
                    correlation_insights.append(
                        f"{color} **{commodity1} vs {commodity2}**: {corr_value:.3f} ({strength})"
                    )
        
        for insight in correlation_insights:
            st.markdown(insight)
            
    except Exception as e:
        st.error(f"""
        ❌ **Error calculating correlations**: {str(e)}
        
        **Debug info:**
        - Commodities: {', '.join(unique_commodities)}
        - Positions: {len(portfolio.positions)}
        
        **Try:**
        - Refreshing the page
        - Adding more diverse positions
        - Checking internet connection
        """)


def risk_analysis_tab(portfolio, analysis_ready):
    """Enhanced risk analysis tab."""
    st.markdown('<div class="section-header">⚠️ Portfolio Risk Analysis</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to see risk metrics")
        return
    
    try:
        portfolio_pnl = portfolio.get_simulation_data()
        
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
            
            # Risk metrics display
            render_comprehensive_risk_metrics(portfolio_pnl)
            
    except Exception as e:
        st.error(f"Could not generate risk analysis: {e}")


def render_comprehensive_risk_metrics(portfolio_pnl):
    st.markdown("### 📊 Comprehensive Risk Metrics (Daily)")
    
    var_95 = np.percentile(portfolio_pnl, 5)
    cvar_95 = np.mean(portfolio_pnl[portfolio_pnl <= var_95])
    prob_loss = np.sum(portfolio_pnl < 0) / len(portfolio_pnl) * 100
    volatility = np.std(portfolio_pnl)
    expected_return = np.mean(portfolio_pnl)
    sharpe_ratio = expected_return / volatility if volatility > 0 else 0
    max_gain = np.max(portfolio_pnl)
    max_loss = np.min(portfolio_pnl)
    
    # First row - DAILY metrics with clear labels
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #FF6B6B;">Daily VaR (95%)</div>
            <div class="metric-value-premium">${var_95:,.0f}</div>
            <div class="metric-subtitle-premium">1-Day Value at Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #dc3545;">Daily CVaR (95%)</div>
            <div class="metric-value-premium">${cvar_95:,.0f}</div>
            <div class="metric-subtitle-premium">1-Day Expected Tail Loss</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #667eea;">Loss Probability</div>
            <div class="metric-value-premium">{prob_loss:.1f}%</div>
            <div class="metric-subtitle-premium">Daily Loss Chance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #4ECDC4;">Daily Volatility</div>
            <div class="metric-value-premium">${volatility:,.0f}</div>
            <div class="metric-subtitle-premium">1-Day Standard Deviation</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row - Expected returns with multiple time frames
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #48bb78;">Daily Expected P&L</div>
            <div class="metric-value-premium">${expected_return:,.0f}</div>
            <div class="metric-subtitle-premium">Average 1-Day Return</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        monthly_expected = expected_return * 21  # 21 trading days per month
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #48bb78;">Monthly Expected P&L</div>
            <div class="metric-value-premium">${monthly_expected:,.0f}</div>
            <div class="metric-subtitle-premium">21-Day Estimate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        annual_expected = expected_return * 252  # 252 trading days per year
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #48bb78;">Annual Expected P&L</div>
            <div class="metric-value-premium">${annual_expected:,.0f}</div>
            <div class="metric-subtitle-premium">252-Day Estimate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #8e44ad;">Daily Sharpe Ratio</div>
            <div class="metric-value-premium">{sharpe_ratio:.3f}</div>
            <div class="metric-subtitle-premium">Risk-Adj. Daily Return</div>
        </div>
        """, unsafe_allow_html=True)


def stress_testing_tab(portfolio, analysis_ready):
    """Enhanced stress testing tab."""
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
        stress_results = run_portfolio_stress_test(portfolio, selected_scenarios)
        
        st.markdown("### 📊 Stress Test Results")
        results_df = pd.DataFrame(stress_results)
        st.dataframe(results_df, use_container_width=True, hide_index=True, height=300)


def run_portfolio_stress_test(portfolio, selected_scenarios):
    """Run stress test on portfolio positions."""
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
            
            # Enhanced hedge calculation for multi-leg strategies
            if position.is_multi_leg:
                # Simplified multi-leg stress calculation
                hedge_pnl = calculate_multi_leg_stress_pnl(position, price_change)
            elif position.strategy == "Futures":
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
    
    return stress_results


def calculate_multi_leg_stress_pnl(position, price_change):
    """Calculate stress P&L for multi-leg strategies."""
    try:
        strategy_type = position.multi_leg_strategy.strategy_type.value
        
        # Get current price to convert position size to contracts
        current_price = position.current_price
        contracts = abs(position.size) / current_price  # Convert dollars to contracts
        
        # Simplified stress calculation based on strategy characteristics
        if "Straddle" in strategy_type:
            # Straddles benefit from volatility
            volatility_benefit = abs(price_change) * contracts * position.hedge_ratio
            if "Long" in strategy_type:
                return volatility_benefit * 0.6  # Long straddles benefit from moves
            else:
                return -volatility_benefit * 0.6  # Short straddles hurt by moves
        
        elif "Strangle" in strategy_type:
            # Similar to straddle but less sensitive
            volatility_benefit = abs(price_change) * contracts * position.hedge_ratio * 0.4
            if "Long" in strategy_type:
                return volatility_benefit
            else:
                return -volatility_benefit
        
        elif "Collar" in strategy_type:
            # Collar provides asymmetric protection
            if price_change < 0:
                return -price_change * contracts * position.hedge_ratio * 0.8
            else:
                return -price_change * contracts * position.hedge_ratio * 0.3
        
        else:
            # Default approximation for other strategies
            return -price_change * contracts * position.hedge_ratio * 0.5
    
    except Exception:
        return 0


# ============================================================================
# SINGLE POSITION INTERFACE WITH MULTI-LEG SUPPORT
# ============================================================================

def render_single_multi_leg_config(commodity, position_size, hedge_ratio):
    """Render multi-leg strategy configuration for single position."""
    st.markdown("**🎯 Multi-Leg Strategy Configuration**")
    
    strategy_type = st.selectbox(
        "Strategy Type:",
        options=[s.value for s in StrategyType],
        help="Select multi-leg options strategy",
        key="single_multi_leg_type"
    )
    
    try:
        current_price = get_current_price(commodity)
    except:
        current_price = 75.0
    
    # Store strategy configuration for analysis
    config = {'strategy_type': strategy_type, 'current_price': current_price}
    
    # Strategy-specific parameters (simplified for single position)
    if strategy_type in ["Long Straddle", "Short Straddle"]:
        strike_price = st.slider(
            "Strike Price ($):",
            min_value=float(current_price * 0.8),
            max_value=float(current_price * 1.2),
            value=float(current_price),
            step=0.5,
            key="single_straddle_strike"
        )
        config['strike_price'] = strike_price
        
    elif strategy_type in ["Long Strangle", "Short Strangle"]:
        col1, col2 = st.columns(2)
        with col1:
            put_strike = st.slider(
                "Put Strike ($):",
                min_value=float(current_price * 0.8),
                max_value=float(current_price * 0.98),
                value=float(current_price * 0.95),
                step=0.5,
                key="single_strangle_put"
            )
        with col2:
            call_strike = st.slider(
                "Call Strike ($):",
                min_value=float(current_price * 1.02),
                max_value=float(current_price * 1.2),
                value=float(current_price * 1.05),
                step=0.5,
                key="single_strangle_call"
            )
        config.update({'put_strike': put_strike, 'call_strike': call_strike})
        
    elif strategy_type == "Collar":
        col1, col2 = st.columns(2)
        with col1:
            put_strike = st.slider(
                "Put Strike ($):",
                min_value=float(current_price * 0.8),
                max_value=float(current_price * 0.98),
                value=float(current_price * 0.90),
                step=0.5,
                key="single_collar_put"
            )
        with col2:
            call_strike = st.slider(
                "Call Strike ($):",
                min_value=float(current_price * 1.02),
                max_value=float(current_price * 1.2),
                value=float(current_price * 1.10),
                step=0.5,
                key="single_collar_call"
            )
        config.update({'put_strike': put_strike, 'call_strike': call_strike})
    
    # Store configuration in session state
    st.session_state['single_multi_leg_config'] = config


def render_single_option_config(commodity):
    """Render single option configuration."""
    st.markdown("**⚙️ Options Parameters:**")
    
    try:
        current_price = get_current_price(commodity)
        strike_price = st.slider(
            "Strike Price ($):",
            min_value=float(current_price * 0.7),
            max_value=float(current_price * 1.3),
            value=float(current_price),
            step=0.5,
            key="single_option_strike"
        )
    except:
        strike_price = st.number_input(
            "Strike Price ($):",
            value=75.0,
            min_value=1.0,
            max_value=200.0,
            step=0.5,
            key="single_option_strike_fallback"
        )
    
    option_type = st.selectbox(
        "Option Type:",
        options=["Put", "Call"],
        index=0,
        key="single_option_type"
    )
    
    # Store option configuration
    st.session_state['single_option_config'] = {
        'strike_price': strike_price,
        'option_type': option_type
    }

def create_multi_leg_position_and_payoff(commodity, position_size, hedge_ratio, current_price):
    """ Create multi-leg position and calculate payoff """
    config = st.session_state.get('single_multi_leg_config', {})
    strategy_type = config.get('strategy_type', 'Long Straddle')
    
    contracts = abs(position_size) / current_price
    
    # Create multi-leg strategy with correct position sizing
    if strategy_type in ["Long Straddle", "Short Straddle"]:
        strike_price = config.get('strike_price', current_price)
        if "Long" in strategy_type:
            multi_leg_strategy = create_long_straddle(
                strike_price, contracts, hedge_ratio, commodity  # Using contracts instead of dollars
            )
        else:
            multi_leg_strategy = create_short_straddle(
                strike_price, contracts, hedge_ratio, commodity  # Using contracts instead of dollars
            )
    
    elif strategy_type in ["Long Strangle", "Short Strangle"]:
        call_strike = config.get('call_strike', current_price * 1.05)
        put_strike = config.get('put_strike', current_price * 0.95)
        if "Long" in strategy_type:
            multi_leg_strategy = create_long_strangle(
                call_strike, put_strike, contracts, hedge_ratio, commodity  # Using contracts instead of dollars
            )
        else:
            multi_leg_strategy = create_short_strangle(
                call_strike, put_strike, contracts, hedge_ratio, commodity  # Using contracts instead of dollars
            )
    
    elif strategy_type == "Collar":
        call_strike = config.get('call_strike', current_price * 1.10)
        put_strike = config.get('put_strike', current_price * 0.90)
        multi_leg_strategy = create_collar(
            call_strike, put_strike, contracts, hedge_ratio, commodity  # Using contracts instead of dollars
        )
    
    elif strategy_type == "Butterfly Spread":
        lower_strike = config.get('lower_strike', current_price * 0.95)
        middle_strike = config.get('middle_strike', current_price)
        upper_strike = config.get('upper_strike', current_price * 1.05)
        multi_leg_strategy = create_butterfly_spread(
            lower_strike, middle_strike, upper_strike, contracts, hedge_ratio, commodity
        )
    
    elif strategy_type == "Iron Condor":
        put_low = config.get('put_strike_low', current_price * 0.85)
        put_high = config.get('put_strike_high', current_price * 0.95)
        call_low = config.get('call_strike_low', current_price * 1.05)
        call_high = config.get('call_strike_high', current_price * 1.15)
        multi_leg_strategy = create_iron_condor(
            put_low, put_high, call_low, call_high, contracts, hedge_ratio, commodity
        )
    
    else:
        # Default to long straddle
        multi_leg_strategy = create_long_straddle(
            current_price, contracts, hedge_ratio, commodity  # Using contracts instead of dollars
        )
    
    # Create position with original dollar size for notional calculations
    position = Position(
        commodity=commodity,
        size=position_size,  # Keep original dollar amount
        hedge_ratio=hedge_ratio,
        strategy="Multi-Leg",
        multi_leg_strategy=multi_leg_strategy
    )
    
    payoff_data = compute_multi_leg_payoff(current_price, multi_leg_strategy)
    
    return position, payoff_data


def create_single_option_position_and_payoff(commodity, position_size, hedge_ratio, current_price):
    """Create single option position and calculate payoff."""
    option_config = st.session_state.get('single_option_config', {})
    strike_price = option_config.get('strike_price', current_price)
    option_type = option_config.get('option_type', 'Put')
    
    position = Position(
        commodity=commodity,
        size=position_size,
        hedge_ratio=hedge_ratio,
        strategy="Options",
        strike_price=strike_price,
        option_type=option_type
    )
    
    payoff_data = compute_payoff_diagram(
        current_price, position_size, hedge_ratio, "Options", strike_price
    )
    
    return position, payoff_data


def render_single_position_summary(position, params, current_price):
    """Render enhanced position summary."""
    st.markdown("### 📋 Position Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium">Commodity</div>
            <div class="metric-value-premium" style="font-size: 1.5rem;">{params['commodity']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium">Current Price</div>
            <div class="metric-value-premium">${current_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        direction = "Long" if params['position_size'] > 0 else "Short"
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium">Position</div>
            <div class="metric-value-premium" style="font-size: 1.3rem;">{direction}</div>
            <div class="metric-subtitle-premium">{abs(params['position_size']):,.0f} units</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        strategy_display = position.strategy_description if position.is_multi_leg else params['strategy']
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium">Hedge Strategy</div>
            <div class="metric-value-premium" style="font-size: 1.3rem;">{strategy_display}</div>
            <div class="metric-subtitle-premium">{params['hedge_ratio']*100:.0f}% hedged</div>
        </div>
        """, unsafe_allow_html=True)


def display_enhanced_payoff_analysis(results):
    """
    Enhanced payoff analysis with comprehensive multi-leg strategy support.
    Shows strategy payoff, underlying P&L, net P&L, and detailed breakdowns.
    """
    payoff_data = results['payoff_data']
    position = results['position']
    current_price = results['current_price']
    
    if position.is_multi_leg:
        display_multi_leg_payoff_visualization(payoff_data, position, current_price, results)
    else:
        display_standard_payoff_visualization(payoff_data, position, current_price, results)


def display_multi_leg_payoff_visualization(payoff_data, position, current_price, results):
    """
    Comprehensive multi-leg strategy payoff visualization.
    Enhanced with strategy insights, Greeks display, and educational content.
    """
    st.markdown("### 🎯 Multi-Leg Strategy Payoff Analysis")
    
    # Strategy header with key information
    render_strategy_header(position, current_price)
    
    # Main payoff chart
    fig_payoff = create_enhanced_multi_leg_chart(payoff_data, position, current_price)
    st.plotly_chart(fig_payoff, use_container_width=True)
    
    # Strategy analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Strategy Breakdown", 
        "📈 Greeks Analysis", 
        "🎓 Strategy Education",
        "⚡ Quick Insights"
    ])
    
    with tab1:
        render_strategy_breakdown_tab(payoff_data, position, current_price)
    
    with tab2:
        render_strategy_greeks_tab(position, current_price)
    
    with tab3:
        render_strategy_education_tab(position)
    
    with tab4:
        render_quick_insights_tab(payoff_data, position, current_price)


def render_strategy_header(position, current_price):
    """Render enhanced strategy header with key metrics."""
    strategy = position.multi_leg_strategy
    
    # Get validation and profile information
    try:
        validation = validate_multi_leg_configuration(strategy, current_price)
        profile = strategy.get_strategy_profile()
    except:
        validation = {'warnings': [], 'estimated_metrics': {}}
        profile = None
    
    # Header with strategy overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #667eea;">Strategy Type</div>
            <div class="metric-value" style="font-size: 1.4rem;">{strategy.strategy_type.value}</div>
            <div class="metric-subtitle-premium">{len(strategy.legs)} legs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        complexity = "High" if len(strategy.legs) > 2 else "Medium"
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #4ECDC4;">Complexity</div>
            <div class="metric-value" style="font-size: 1.4rem;">{complexity}</div>
            <div class="metric-subtitle-premium">Skill Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        estimated_premium = validation.get('estimated_metrics', {}).get('total_premium', 0)
        premium_type = "Debit" if estimated_premium > 0 else "Credit" if estimated_premium < 0 else "Even"
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #48bb78;">Net Premium</div>
            <div class="metric-value" style="font-size: 1.4rem;">${abs(estimated_premium):.2f}</div>
            <div class="metric-subtitle-premium">{premium_type}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        hedge_effectiveness = f"{strategy.hedge_ratio:.1%}"
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #FF6B6B;">Hedge Ratio</div>
            <div class="metric-value" style="font-size: 1.4rem;">{hedge_effectiveness}</div>
            <div class="metric-subtitle-premium">Coverage</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display warnings if any
    if validation.get('warnings'):
        st.markdown("### ⚠️ Configuration Alerts")
        for warning in validation['warnings'][:3]:  # Show max 3 warnings
            st.warning(f"⚠️ {warning}")


def create_enhanced_multi_leg_chart(payoff_data, position, current_price):
    """Create enhanced multi-leg strategy payoff chart with detailed annotations."""
    fig = go.Figure()
    
    # Extract data
    spot_prices = payoff_data['spot_prices']
    underlying_pnl = payoff_data['underlying_pnl']
    net_pnl = payoff_data['net_pnl']
    
    # Add underlying P&L (dashed line)
    fig.add_trace(go.Scatter(
        x=spot_prices,
        y=underlying_pnl,
        mode='lines',
        name='Underlying P&L',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        opacity=0.7,
        hovertemplate='<b>Underlying P&L</b><br>Price: $%{x:.2f}<br>P&L: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add strategy payoff if available
    if 'strategy_payoff' in payoff_data:
        fig.add_trace(go.Scatter(
            x=spot_prices,
            y=payoff_data['strategy_payoff'],
            mode='lines',
            name=f'{position.multi_leg_strategy.strategy_type.value} Payoff',
            line=dict(color='#9c27b0', width=2, dash='dot'),
            opacity=0.8,
            hovertemplate='<b>Strategy Payoff</b><br>Price: $%{x:.2f}<br>Payoff: $%{y:,.0f}<extra></extra>'
        ))
    
    # Add net strategy payoff if available
    if 'net_strategy_payoff' in payoff_data:
        fig.add_trace(go.Scatter(
            x=spot_prices,
            y=payoff_data['net_strategy_payoff'],
            mode='lines',
            name='Net Strategy (After Premium)',
            line=dict(color='#667eea', width=3),
            hovertemplate='<b>Net Strategy</b><br>Price: $%{x:.2f}<br>P&L: $%{y:,.0f}<extra></extra>'
        ))
    
    # Add total net P&L (thick green line)
    fig.add_trace(go.Scatter(
        x=spot_prices,
        y=net_pnl,
        mode='lines',
        name='Total Net P&L',
        line=dict(color='#48bb78', width=4),
        hovertemplate='<b>Total Net P&L</b><br>Price: $%{x:.2f}<br>P&L: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add breakeven lines
    if 'breakeven_prices' in payoff_data and payoff_data['breakeven_prices']:
        for i, breakeven in enumerate(payoff_data['breakeven_prices']):
            fig.add_vline(
                x=breakeven,
                line_dash="dot",
                line_color="orange",
                line_width=2,
                annotation_text=f"BE {i+1}: ${breakeven:.2f}",
                annotation_position="top",
                annotation_font_size=10
            )
    
    # Add current price line
    fig.add_vline(
        x=current_price,
        line_dash="solid",
        line_color="blue",
        line_width=2,
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="bottom",
        annotation_font_color="blue",
        annotation_font_size=12
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
    
    # Enhanced layout
    fig.update_layout(
        title={
            'text': f"{position.multi_leg_strategy.strategy_type.value} - Payoff at Expiration",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter', 'color': '#2d3748'}
        },
        xaxis_title="Underlying Price ($)",
        yaxis_title="Profit & Loss ($)",
        height=500,
        font=dict(family="Inter", size=12),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    return fig


def render_strategy_breakdown_tab(payoff_data, position, current_price):
    """Render detailed strategy breakdown with legs analysis."""
    st.markdown("### 📋 Strategy Composition")
    
    strategy = position.multi_leg_strategy
    
    # Legs breakdown table
    legs_data = []
    for i, leg in enumerate(strategy.legs):
        # Calculate moneyness
        moneyness = current_price / leg.strike_price
        if leg.option_type.lower() == 'call':
            if moneyness > 1.05:
                money_status = "ITM"
            elif moneyness > 0.95:
                money_status = "ATM"
            else:
                money_status = "OTM"
        else:  # put
            if moneyness < 0.95:
                money_status = "ITM"
            elif moneyness < 1.05:
                money_status = "ATM"
            else:
                money_status = "OTM"
        
        legs_data.append({
            'Leg': f'Leg {i+1}',
            'Type': leg.option_type.title(),
            'Strike': f'${leg.strike_price:.2f}',
            'Position': 'Long' if leg.quantity > 0 else 'Short',
            'Quantity': abs(leg.quantity),
            'Moneyness': money_status,
            'Expiry': f'{leg.expiry_months}M'
        })
    
    legs_df = pd.DataFrame(legs_data)
    st.dataframe(legs_df, use_container_width=True, hide_index=True)
    
    # Strategy metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Strategy Metrics:**")
        
        # Extract key metrics from payoff data
        max_profit = payoff_data.get('max_profit', 'Unlimited')
        max_loss = payoff_data.get('max_loss', 'Limited')
        total_premium = payoff_data.get('total_premium', 0)
        
        st.write(f"• **Max Profit:** ${max_profit:.2f}" if isinstance(max_profit, (int, float)) else f"• **Max Profit:** {max_profit}")
        st.write(f"• **Max Loss:** ${abs(max_loss):.2f}" if isinstance(max_loss, (int, float)) else f"• **Max Loss:** {max_loss}")
        st.write(f"• **Net Premium:** ${total_premium:.2f} {'(Debit)' if total_premium > 0 else '(Credit)' if total_premium < 0 else '(Even)'}")
        
        # Breakeven information
        if 'breakeven_prices' in payoff_data and payoff_data['breakeven_prices']:
            breakevens = payoff_data['breakeven_prices']
            st.write(f"• **Breakevens:** {len(breakevens)} points")
            for i, be in enumerate(breakevens):
                st.write(f"  - BE {i+1}: ${be:.2f}")
        else:
            st.write("• **Breakevens:** Not calculated")
    
    with col2:
        st.markdown("**🎯 Strategy Profile:**")
        
        try:
            profile = strategy.get_strategy_profile()
            if profile:
                st.write(f"• **Best For:** {profile.get('Best For', 'N/A')}")
                st.write(f"• **Market Outlook:** {profile.get('Market Outlook', 'N/A')}")
                st.write(f"• **Volatility View:** {profile.get('Volatility View', 'N/A')}")
                st.write(f"• **Time Decay:** {profile.get('Time Decay', 'N/A')}")
                st.write(f"• **Complexity:** {profile.get('Complexity', 'N/A')}")
            else:
                st.write("• Profile information not available")
        except:
            st.write("• Profile information not available")


def render_strategy_greeks_tab(position, current_price):
    """Render comprehensive Greeks analysis for multi-leg strategy."""
    st.markdown("### 📈 Greeks Analysis")
    
    try:
        strategy = position.multi_leg_strategy
        greeks = strategy.get_strategy_greeks(current_price)
        
        # Main Greeks display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta_color = "#4ECDC4" if greeks['delta'] >= 0 else "#FF6B6B"
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {delta_color};">Delta</div>
                <div class="metric-value-premium">{greeks['delta']:.4f}</div>
                <div class="metric-subtitle-premium">Price Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: #48bb78;">Gamma</div>
                <div class="metric-value-premium">{greeks['gamma']:.6f}</div>
                <div class="metric-subtitle-premium">Delta Change</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            theta_color = "#FF6B6B" if greeks['theta'] < 0 else "#48bb78"
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: {theta_color};">Theta</div>
                <div class="metric-value-premium">${greeks['theta']:.2f}</div>
                <div class="metric-subtitle-premium">Time Decay/Day</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: #667eea;">Vega</div>
                <div class="metric-value-premium">${greeks['vega']:.2f}</div>
                <div class="metric-subtitle-premium">Vol Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card-premium">
                <div class="metric-title-premium" style="color: #8e44ad;">Rho</div>
                <div class="metric-value-premium">${greeks['rho']:.2f}</div>
                <div class="metric-subtitle-premium">Rate Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Greeks interpretation
        st.markdown("### 🧠 Greeks Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Risk Assessment:**")
            
            # Delta analysis
            if abs(greeks['delta']) > 0.5:
                st.warning(f"🔸 High directional exposure (Δ = {greeks['delta']:.3f})")
            elif abs(greeks['delta']) > 0.2:
                st.info(f"🔸 Moderate directional bias (Δ = {greeks['delta']:.3f})")
            else:
                st.success(f"🔸 Delta neutral strategy (Δ = {greeks['delta']:.3f})")
            
            # Theta analysis
            if greeks['theta'] < -50:
                st.warning(f"🔸 High time decay (Θ = ${greeks['theta']:.2f}/day)")
            elif greeks['theta'] < -10:
                st.info(f"🔸 Moderate time decay (Θ = ${greeks['theta']:.2f}/day)")
            else:
                st.success(f"🔸 Low time decay impact (Θ = ${greeks['theta']:.2f}/day)")
        
        with col2:
            st.markdown("**📈 Market Sensitivity:**")
            
            # Vega analysis
            if abs(greeks['vega']) > 100:
                st.warning(f"🔸 High volatility risk (ν = ${greeks['vega']:.2f})")
            elif abs(greeks['vega']) > 50:
                st.info(f"🔸 Moderate volatility sensitivity (ν = ${greeks['vega']:.2f})")
            else:
                st.success(f"🔸 Low volatility impact (ν = ${greeks['vega']:.2f})")
    
    except Exception as e:
        st.error(f"Could not calculate Greeks: {e}")


def render_strategy_education_tab(position):
    """Render educational content for the strategy."""
    st.markdown("### 🎓 Strategy Education")
    
    strategy_name = position.multi_leg_strategy.strategy_type.value
    
    # Educational content map
    education_map = {
        "Long Straddle": {
            "when_to_use": "Before major events like earnings, FDA approvals, or OPEC meetings when high volatility is expected",
            "profit_mechanism": "Profits when the actual price movement exceeds the implied volatility priced into the options",
            "risk_factors": "Time decay accelerates as expiration approaches, especially harmful if volatility doesn't materialize",
            "management_tips": "Consider closing at 50% profit or 25% loss. Roll to later expiration if volatility expansion is delayed",
            "example_scenario": "Oil at $75 before OPEC meeting - buy $75 calls and puts expecting 10%+ move in either direction"
        },
        "Collar": {
            "when_to_use": "When holding a long commodity position and wanting downside protection while generating income",
            "profit_mechanism": "Put provides floor protection, call premium reduces the cost of protection",
            "risk_factors": "Limited upside participation above the call strike price",
            "management_tips": "Roll strikes higher in bull markets, consider closing before assignment",
            "example_scenario": "Own 1000 barrels at $70, buy $65 puts, sell $80 calls for protection with income"
        },
        "Long Strangle": {
            "when_to_use": "When expecting significant price movement but uncertain of direction, lower cost than straddle",
            "profit_mechanism": "Profits from large moves in either direction, needs move beyond both strikes plus premium",
            "risk_factors": "Requires larger moves than straddle to profit, time decay is still significant",
            "management_tips": "Close early if reaching 50% profit, roll strikes closer if expecting continued volatility",
            "example_scenario": "Oil at $75, buy $70 puts and $80 calls expecting 8%+ move from geopolitical events"
        }
    }
    
    content = education_map.get(strategy_name, {})
    
    if content:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📚 When to Use:**")
            st.write(content.get('when_to_use', 'Information not available'))
            
            st.markdown("**💰 Profit Mechanism:**")
            st.write(content.get('profit_mechanism', 'Information not available'))
        
        with col2:
            st.markdown("**⚠️ Risk Factors:**")
            st.write(content.get('risk_factors', 'Information not available'))
            
            st.markdown("**🔧 Management Tips:**")
            st.write(content.get('management_tips', 'Information not available'))
        
        st.markdown("**🎯 Example Scenario:**")
        st.info(content.get('example_scenario', 'Information not available'))
    else:
        st.info(f"Educational content for {strategy_name} is being developed.")


def render_quick_insights_tab(payoff_data, position, current_price):
    """Render quick insights and key takeaways."""
    st.markdown("### ⚡ Quick Insights")
    
    strategy_name = position.multi_leg_strategy.strategy_type.value
    
    # Strategy-specific insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Key Takeaways:**")
        if "Long Straddle" in strategy_name:
            st.write("• Profits from large price movements in either direction")
            st.write("• Maximum profit is unlimited")
            st.write("• Best deployed before high-volatility events")
        elif "Collar" in strategy_name:
            st.write("• Provides downside protection for long positions") 
            st.write("• Call premium helps fund put protection")
            st.write("• Good for volatile markets with directional bias")
        elif "Long Strangle" in strategy_name:
            st.write("• Lower cost alternative to straddle")
            st.write("• Profits from large moves in either direction")
            st.write("• Wider breakeven range than straddle")
        else:
            st.write("• Multi-leg options strategy")
            st.write("• Complex risk/reward profile")
            st.write("• Requires active management")
    
    with col2:
        st.markdown("**⚠️ Watch Out For:**")
        if "Long Straddle" in strategy_name:
            st.write("• Time decay accelerates near expiration")
            st.write("• Requires significant price movement")
            st.write("• Volatility crush after events")
        elif "Collar" in strategy_name:
            st.write("• Limits upside participation")
            st.write("• May be assigned on call if price rises")
            st.write("• Protection level depends on strikes")
        elif "Long Strangle" in strategy_name:
            st.write("• Requires larger moves than straddle")
            st.write("• Time decay still significant")
            st.write("• Premium loss if volatility doesn't materialize")
        else:
            st.write("• Strategy-specific risks apply")
            st.write("• Monitor market conditions closely")
            st.write("• Consider exit strategies")
    
    # Scenario analysis
    st.markdown("### 📊 Scenario Analysis")
    
    scenarios = [
        {"name": "📈 Strong Rally (+15%)", "outcome": "🟢 Likely Profit" if "Long" in strategy_name else "🟡 Depends"},
        {"name": "📉 Sharp Decline (-15%)", "outcome": "🟢 Likely Profit" if "Long" in strategy_name else "🟡 Depends"},
        {"name": "🔄 Sideways (±2%)", "outcome": "🔴 Likely Loss" if "Long" in strategy_name else "🟢 Likely Profit"},
        {"name": "💥 High Volatility", "outcome": "🟢 Benefits Strategy" if "Long" in strategy_name else "🔴 Hurts Strategy"}
    ]
    
    for scenario in scenarios:
        st.write(f"{scenario['name']}: {scenario['outcome']}")


def display_standard_payoff_visualization(payoff_data, position, current_price, results):
    """Enhanced standard payoff visualization for futures and single options."""
    st.markdown("### 📊 Single Strategy Payoff Analysis")
    
    # Create enhanced standard chart
    fig_payoff = go.Figure()
    
    spot_prices = payoff_data['spot_prices']
    underlying_pnl = payoff_data['underlying_pnl']
    hedge_pnl = payoff_data['hedge_pnl']
    net_pnl = payoff_data['net_pnl']
    
    # Add traces
    fig_payoff.add_trace(go.Scatter(
        x=spot_prices, y=underlying_pnl, mode='lines', name='Underlying P&L',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    fig_payoff.add_trace(go.Scatter(
        x=spot_prices, y=hedge_pnl, mode='lines', name='Hedge P&L',
        line=dict(color='#4ECDC4', width=2, dash='dot')
    ))
    
    fig_payoff.add_trace(go.Scatter(
        x=spot_prices, y=net_pnl, mode='lines', name='Net P&L',
        line=dict(color='#48bb78', width=4)
    ))
    
    # Add lines
    fig_payoff.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    fig_payoff.add_vline(x=current_price, line_dash="dash", line_color="orange")
    
    # Layout
    strategy_name = position.strategy
    if position.strategy == "Options":
        strategy_name += f" {position.option_type}"
    
    fig_payoff.update_layout(
        title=f"{strategy_name} Strategy - Payoff Analysis",
        xaxis_title="Underlying Price ($)",
        yaxis_title="Profit & Loss ($)",
        height=450,
        font=dict(family="Inter")
    )
    
    st.plotly_chart(fig_payoff, use_container_width=True)


def display_strategy_legs_table(multi_leg_strategy):
    """Display strategy legs breakdown table."""
    st.markdown("### 📋 Strategy Composition")
    
    legs_data = []
    for i, leg in enumerate(multi_leg_strategy.legs):
        legs_data.append({
            'Leg': f'Leg {i+1}',
            'Type': leg.option_type.title(),
            'Strike': f'${leg.strike_price:.2f}',
            'Position': 'Long' if leg.quantity > 0 else 'Short',
            'Quantity': abs(leg.quantity),
            'Expiry': f'{leg.expiry_months} months'
        })
    
    legs_df = pd.DataFrame(legs_data)
    st.dataframe(legs_df, use_container_width=True, hide_index=True)


def display_enhanced_risk_metrics(results):
    """Display enhanced risk metrics comparison."""
    hedged_risk = results['hedged_risk']
    unhedged_risk = results['unhedged_risk']
    
    risk_comparison = summarize_risk_comparison(hedged_risk, unhedged_risk)
    
    st.markdown("### 📊 Risk Metrics Comparison")
    st.dataframe(risk_comparison, use_container_width=True, hide_index=True)
    
    effectiveness = compare_hedging_effectiveness(
        results['sim_results']['hedged_pnl'], 
        results['sim_results']['unhedged_pnl']
    )
    
    st.markdown("### 🎯 Hedging Effectiveness")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #4ECDC4;">Volatility Reduction</div>
            <div class="metric-value-premium">{effectiveness['volatility_reduction']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #48bb78;">Loss Prob. Reduction</div>
            <div class="metric-value-premium">{effectiveness['loss_prob_reduction']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #667eea;">Expected P&L Change</div>
            <div class="metric-value-premium">${effectiveness['mean_difference']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-title-premium" style="color: #8e44ad;">Sharpe Improvement</div>
            <div class="metric-value-premium">{effectiveness['sharpe_improvement']:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)


def display_single_stress_tests(results):
    """Display stress test results for single position."""
    params = results['params']
    current_price = results['current_price']
    position = results['position']
    
    st.markdown("### 🧪 Historical Crisis Stress Testing")
    
    selected_scenarios = st.multiselect(
        "Select crisis scenarios:",
        options=list(STRESS_SCENARIOS.keys()),
        default=list(STRESS_SCENARIOS.keys())[:3],
        key="single_stress_scenarios"
    )
    
    if selected_scenarios:
        stress_results = []
        
        for scenario_name in selected_scenarios:
            scenario = STRESS_SCENARIOS[scenario_name]
            price_change = scenario["oil_peak_to_trough"]
            
            position_size = abs(params['position_size'])
            price_shock = current_price * price_change
            
            unhedged_pnl = price_shock * position_size
            
            # Enhanced hedge calculation for different strategies
            hedge_ratio = params['hedge_ratio']
            
            if position.is_multi_leg:
                hedge_pnl = calculate_multi_leg_stress_pnl(position, price_shock)
            elif params['strategy'] == "Futures":
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


def display_strategy_details_tab(position):
    """Display detailed strategy information tab."""
    if position.is_multi_leg:
        st.markdown("### 🎯 Multi-Leg Strategy Details")
        
        strategy = position.multi_leg_strategy
        
        # Strategy overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Strategy Information:**")
            st.write(f"• **Type:** {strategy.strategy_type.value}")
            st.write(f"• **Legs:** {len(strategy.legs)}")
            st.write(f"• **Underlying Size:** {strategy.underlying_size:,.0f}")
            st.write(f"• **Hedge Ratio:** {strategy.hedge_ratio:.1%}")
            st.write(f"• **Commodity:** {strategy.commodity}")
        
        with col2:
            st.markdown("**📝 Strategy Description:**")
            st.write(strategy.description)
            
            # Estimate total premium
            try:
                current_price = position.current_price
                total_premium = strategy.get_total_premium(current_price)
                st.write(f"• **Estimated Premium:** ${total_premium:.2f}")
            except:
                st.write("• **Estimated Premium:** Not available")
        
        # Detailed legs breakdown
        st.markdown("### 📋 Individual Legs Breakdown")
        display_strategy_legs_table(strategy)
        
        # Greeks information if available
        try:
            greeks = strategy.get_strategy_greeks(position.current_price)
            st.markdown("### 📈 Strategy Greeks")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Delta", f"{greeks['delta']:.4f}")
            with col2:
                st.metric("Gamma", f"{greeks['gamma']:.6f}")
            with col3:
                st.metric("Theta", f"${greeks['theta']:.2f}")
            with col4:
                st.metric("Vega", f"${greeks['vega']:.2f}")
            with col5:
                st.metric("Rho", f"${greeks['rho']:.2f}")
                
        except Exception as e:
            st.info("Greeks calculation not available for this strategy")
    
    elif position.strategy == "Options":
        st.markdown("### 📈 Single Option Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Option Information:**")
            st.write(f"• **Type:** {position.option_type}")
            st.write(f"• **Strike Price:** ${position.strike_price:.2f}")
            st.write(f"• **Current Price:** ${position.current_price:.2f}")
            
            moneyness = position.current_price / position.strike_price
            if position.option_type.lower() == 'call':
                if moneyness > 1.05:
                    st.write("• **Moneyness:** In-the-Money")
                elif moneyness > 0.95:
                    st.write("• **Moneyness:** At-the-Money")
                else:
                    st.write("• **Moneyness:** Out-of-the-Money")
            else:  # put
                if moneyness < 0.95:
                    st.write("• **Moneyness:** In-the-Money")
                elif moneyness < 1.05:
                    st.write("• **Moneyness:** At-the-Money")
                else:
                    st.write("• **Moneyness:** Out-of-the-Money")
        
        with col2:
            st.markdown("**📝 Option Characteristics:**")
            if position.option_type.lower() == 'call':
                st.write("• **Direction:** Bullish protection")
                st.write("• **Max Loss:** Premium paid")
                st.write("• **Max Gain:** Unlimited")
                st.write("• **Best for:** Protecting short positions")
            else:
                st.write("• **Direction:** Bearish protection")
                st.write("• **Max Loss:** Premium paid")
                st.write("• **Max Gain:** Strike - Premium")
                st.write("• **Best for:** Protecting long positions")
    
    else:  # Futures
        st.markdown("### 🔄 Futures Hedge Details")
        
        st.markdown("**📊 Futures Information:**")
        st.write(f"• **Strategy:** Linear hedge")
        st.write(f"• **Hedge Ratio:** {position.hedge_ratio:.1%}")
        st.write(f"• **Position Size:** {abs(position.size):,.0f}")
        st.write(f"• **Current Price:** ${position.current_price:.2f}")
        
        st.markdown("**📝 Futures Characteristics:**")
        st.write("• **Direction:** Symmetric protection")
        st.write("• **Max Loss:** Unlimited")
        st.write("• **Max Gain:** Unlimited")
        st.write("• **Best for:** Linear price exposure management")
        st.write("• **Correlation:** High with underlying commodity")


# ============================================================================
# PORTFOLIO INTERFACE AND BUILDER
# ============================================================================

def portfolio_interface():
    """Enhanced portfolio management interface with centered buttons"""
    
    # Initialize sidebar state
    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = False
    
    # Create main layout
    if st.session_state.sidebar_open:
        # Sidebar is open - use columns
        col_sidebar, col_main = st.columns([1, 2])
        
        with col_sidebar:
            # Sidebar toggle button (close)
            if st.button("✖️ Close Controls", type="secondary", use_container_width=True):
                st.session_state.sidebar_open = False
                st.rerun()
            
            st.markdown("---")
            portfolio_builder_sidebar()
        
        with col_main:
            portfolio_dashboard()
    
    else:
        # Sidebar is closed - show CENTERED action buttons
        portfolio = st.session_state.portfolio_manager
        
        # ALWAYS show centered buttons (whether portfolio empty or not)
        render_centered_action_buttons_with_summary(portfolio)
        
        st.markdown("---")
        
        # Full-width dashboard
        portfolio_dashboard()


def render_centered_action_buttons_with_summary(portfolio):
    """Render centered action buttons with portfolio summary when needed"""
    
    # CENTERED ACTION BUTTONS
    st.markdown("""
    <style>
    .centered-action-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered columns for buttons
    col1, col2, col3, col4, col5 = st.columns([2.5, 1, 1, 1, 2.5])
    
    with col2:
        if st.button("➕ Add Position", type="primary", use_container_width=True):
            st.session_state.sidebar_open = True
            st.rerun()
    
    with col3:
        if st.button("📝 Load Sample", type="secondary", use_container_width=True):
            st.session_state.portfolio_manager = create_sample_portfolio()
            st.success("✅ Sample portfolio loaded!")
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            st.session_state.portfolio_manager.clear()
            st.success("✅ Portfolio cleared!")
            st.rerun()

def render_enhanced_portfolio_summary(portfolio):
    """Portfolio summary using Streamlit metrics (reliable)"""
    
    if len(portfolio.positions) == 0:
        return
    
    # Calculate metrics
    total_notional = 0
    long_notional = 0
    short_notional = 0
    
    for position in portfolio.positions.values():
        notional = position.notional_value
        total_notional += notional
        
        if position.size > 0:
            long_notional += notional
        else:
            short_notional += abs(notional)
    
    net_exposure = long_notional - short_notional
    position_count = len(portfolio.positions)
    commodities = len(set(pos.commodity for pos in portfolio.positions.values()))
    
    # Display summary
    st.markdown("### 📊 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Portfolio", f"${total_notional:,.0f}", f"{position_count} positions")
    
    with col2:
        long_pct = (long_notional/total_notional*100) if total_notional > 0 else 0
        st.metric("Long Exposure", f"${long_notional:,.0f}", f"{long_pct:.1f}%")
    
    with col3:
        short_pct = (short_notional/total_notional*100) if total_notional > 0 else 0
        st.metric("Short Exposure", f"${short_notional:,.0f}", f"{short_pct:.1f}%")
    
    with col4:
        bias = "Long bias" if net_exposure > 0 else "Short bias" if net_exposure < 0 else "Neutral"
        st.metric("Net Exposure", f"${net_exposure:+,.0f}", bias)
    
    st.markdown("---")

def portfolio_builder_sidebar():
    """Enhanced portfolio builder sidebar with better organization"""
    
    portfolio = st.session_state.portfolio_manager
    
    # Portfolio overview (always visible)
    if len(portfolio) > 0:
        total_notional = sum(pos.notional_value for pos in portfolio.positions.values())
        unique_commodities = len(set(pos.commodity for pos in portfolio.positions.values()))
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 1rem 0;">📊 Portfolio Overview</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: bold;">{len(portfolio)}</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">Positions</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: bold;">{unique_commodities}</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">Commodities</div>
                </div>
            </div>
            <div style="text-align: center; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
                <div style="font-size: 1.4rem; font-weight: bold;">${total_notional:,.0f}</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">Total Notional</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions section
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Load Sample", type="secondary", use_container_width=True, key="sidebar_load"):
            st.session_state.portfolio_manager = create_sample_portfolio()
            st.success("✅ Sample loaded!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True, key="sidebar_clear"):
            st.session_state.portfolio_manager.clear()
            st.success("✅ Cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Position addition form (enhanced)
    render_enhanced_portfolio_position_form()
    
    st.markdown("---")
    
    # Current positions (if any)
    if len(portfolio) > 0:
        render_current_positions_list(portfolio)
        st.markdown("---")
    
    # Analysis settings
    render_portfolio_analysis_settings()
    
    st.markdown("---")
    
    # Analysis button
    analysis_disabled = len(portfolio) == 0
    if st.button("🚀 Run Portfolio Analysis", 
                 type="primary", 
                 use_container_width=True, 
                 disabled=analysis_disabled,
                 key="sidebar_analyze"):
        if len(portfolio) > 0:
            st.session_state.simulation_run = True
            # Close sidebar after running analysis
            st.session_state.sidebar_open = False
            st.rerun()


def render_enhanced_portfolio_position_form():
    """Enhanced position form with better UX and spacing"""
    
    st.markdown("### ➕ Add New Position")
    
    # Initialize form state
    if 'form_errors' not in st.session_state:
        st.session_state.form_errors = []
    
    # Position name with validation
    position_name = st.text_input(
        "Position Name:",
        placeholder="e.g., 'oil_hedge_q4'",
        help="Unique identifier for this position",
        key="new_position_name"
    )
    
    # Real-time validation
    name_error = None
    if position_name:
        if position_name in st.session_state.portfolio_manager.positions:
            name_error = "⚠️ Name already exists"
            st.error(name_error)
        elif len(position_name) < 3:
            name_error = "⚠️ Name too short (min 3 chars)"
            st.error(name_error)
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # Commodity selection
    commodity = render_enhanced_commodity_selector("new_position")
    
    # Show commodity specifications
    render_commodity_info_panel(commodity)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Strategy selection first (affects other inputs) 
    strategy = st.selectbox(
        "Hedging Strategy:",
        options=["Futures", "Options", "Multi-Leg Options"],
        help="Type of hedging instrument",
        key="new_strategy"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Get commodity specs for dynamic sizing
    from hedging.data import get_commodity_specs
    specs = get_commodity_specs(commodity)
    typical_size = specs.get('contract_size', 1000)
    
    # Direction & Size based on strategy
    if strategy == "Multi-Leg Options":
        # Multi-leg: only size (direction from strategy type)
        position_size = st.number_input(
            f"Position Size ({specs['unit']}):",
            min_value=1.0,
            max_value=100000.0,
            value=float(typical_size),
            step=float(typical_size/10),
            help="Underlying exposure to hedge",
            key="new_size"
        )
        position_direction = "Long"  # Default
        
        multi_leg_type = st.session_state.get('multi_leg_strategy_type', 'Long Straddle')
        st.caption(f"🎯 {multi_leg_type}: {position_size:,.0f} units")
        
    else:
        # Regular strategies: direction + size
        col_dir, col_size = st.columns([1, 1.5])
        
        with col_dir:
            position_direction = st.selectbox(
                "Direction:",
                options=["Long", "Short"],
                key="new_direction"
            )
        
        with col_size:
            position_size = st.number_input(
                f"Size ({specs['unit']}):",
                min_value=1.0,
                max_value=100000.0,
                value=float(typical_size),
                step=float(typical_size/10),
                key="new_size"
            )
        
        direction_emoji = "📈" if position_direction == "Long" else "📉"
        st.caption(f"{direction_emoji} {position_direction} {position_size:,.0f} units")
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # Strategy-specific parameters
    if strategy == "Multi-Leg Options":
        render_multi_leg_strategy_selector()
    elif strategy == "Options":
        render_single_option_parameters()
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # Hedge ratio
    st.markdown("**Risk Management:**")
    hedge_ratio = st.slider(
        "Hedge Ratio:",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=5.0,
        format="%.0f%%",
        help="Percentage of position to hedge",
        key="new_hedge_ratio"
    ) / 100.0
    
    # Visual hedge indicator
    if hedge_ratio >= 0.8:
        hedge_status = "🛡️ High Protection"
        hedge_color = "#28a745"
    elif hedge_ratio >= 0.5:
        hedge_status = "⚠️ Moderate Protection"
        hedge_color = "#ffc107"
    else:
        hedge_status = "🚫 Low Protection"
        hedge_color = "#dc3545"
    
    st.markdown(f"""
    <div style="background: {hedge_color}; color: white; padding: 0.4rem; 
                border-radius: 6px; text-align: center; font-size: 0.85rem; margin: 0.5rem 0;">
        {hedge_status} ({hedge_ratio:.0%})
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # Add position button
    button_disabled = bool(name_error) or not position_name
    
    if st.button("🚀 Add Position", 
                 type="primary" if not button_disabled else "secondary",
                 disabled=button_disabled,
                 use_container_width=True,
                 key="add_position_btn"):
        
        try:
            # Adjust size for direction
            if position_direction == "Short":
                position_size = -position_size
            
            # Create position
            new_position = create_position_from_form(
                position_name, commodity, position_size, hedge_ratio, strategy
            )
            
            # Add to portfolio
            st.session_state.portfolio_manager.add_position(position_name, new_position)
            
            st.success(f"✅ Added {position_name}!")
            
            # Clear form
            for key in ['new_position_name', 'new_commodity', 'new_direction', 'new_size', 'new_strategy']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def add_sidebar_spacing_css():
    """Add custom CSS for better sidebar spacing"""
    st.markdown("""
    <style>
    /* Better sidebar spacing */
    .css-1d391kg {
        padding-top: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    
    /* More space between main content and sidebar */
    .css-18e3th9 {
        padding-left: 2rem;
    }
    
    /* Better button spacing */
    .stButton > button {
        margin-bottom: 0.5rem;
    }
    
    /* Better form spacing */
    .stSelectbox, .stTextInput, .stNumberInput {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_strategy_summary_box(strategy, hedge_ratio, commodity, position_size):
    """Render a summary box showing the current strategy configuration"""
    
    # Get strategy-specific information
    if strategy == "Multi-Leg Options" and st.session_state.get('multi_leg_config'):
        config = st.session_state.multi_leg_config
        strategy_desc = config.get('strategy_type', 'Multi-Leg')
        strategy_detail = f"{strategy_desc} Strategy"
        
        # Add estimated cost if available
        try:
            current_price = get_current_price(commodity)
            estimated_cost = estimate_strategy_cost(strategy_desc, current_price, 
                                                  config.get('strike_price', current_price))
            cost_info = f"Est. Cost: ${estimated_cost:.2f}"
        except:
            cost_info = "Cost: TBD"
            
    elif strategy == "Options":
        strike_price = st.session_state.get('portfolio_strike_price', 75.0)
        option_type = st.session_state.get('portfolio_option_type', 'Put')
        strategy_detail = f"{option_type} Option"
        cost_info = f"Strike: ${strike_price:.2f}"
        
    else:  # Futures
        strategy_detail = "Futures Hedge"
        cost_info = "Linear Protection"
    
    # Calculate estimated notional
    try:
        current_price = get_current_price(commodity)
        notional = position_size * current_price
        notional_info = f"Notional: ${notional:,.0f}"
    except:
        notional_info = "Notional: TBD"
    
    # Render summary box
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%); 
                padding: 1rem; border-radius: 12px; border-left: 4px solid #667eea; margin: 1rem 0;">
        <div style="font-weight: 600; color: #2d3748; margin-bottom: 0.5rem; font-size: 1rem;">
            📊 Strategy Preview
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
            <div><strong>Strategy:</strong> {strategy_detail}</div>
            <div><strong>Hedge:</strong> {hedge_ratio:.0%}</div>
            <div><strong>{cost_info}</strong></div>
            <div><strong>{notional_info}</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def handle_add_position(position_name, commodity, position_direction, position_size, strategy, hedge_ratio):
    """Handle the position addition with proper error handling and UX feedback"""
    
    # Set loading state
    st.session_state.adding_position = True
    st.session_state.form_errors = []
    
    try:
        # Adjust position size based on direction
        if position_direction == "Short":
            position_size = -position_size
        
        # Create the position
        new_position = create_position_from_form(
            position_name, commodity, position_size, hedge_ratio, strategy
        )
        
        # Add to portfolio
        st.session_state.portfolio_manager.add_position(position_name, new_position)
        
        # Clear form state
        clear_form_state()
        
        # Show success message
        st.success(f"✅ Added {position_name} to portfolio!")
        
        # Small delay for UX
        import time
        time.sleep(0.5)
        
        st.rerun()
        
    except Exception as e:
        st.session_state.form_errors = [f"Error creating position: {str(e)}"]
        st.session_state.adding_position = False
        st.error(f"❌ Failed to add position: {str(e)}")
        
    finally:
        st.session_state.adding_position = False


def clear_form_state():
    """Clear form-related session state"""
    form_keys = [
        'new_position_name', 'new_commodity', 'new_direction', 'new_size',
        'new_strategy', 'new_hedge_ratio', 'form_errors'
    ]
    
    for key in form_keys:
        if key in st.session_state:
            del st.session_state[key]


# Enhanced validation function
def validate_position_inputs(position_name, commodity, position_size, hedge_ratio):
    """Comprehensive input validation with user-friendly error messages"""
    errors = []
    
    # Position name validation
    if not position_name:
        errors.append("Position name is required")
    elif len(position_name) < 3:
        errors.append("Position name must be at least 3 characters")
    elif position_name in st.session_state.portfolio_manager.positions:
        errors.append("Position name already exists")
    elif not position_name.replace('_', '').replace('-', '').isalnum():
        errors.append("Position name can only contain letters, numbers, hyphens, and underscores")
    
    # Position size validation
    if position_size <= 0:
        errors.append("Position size must be positive")
    elif position_size > 100000:
        errors.append("Position size cannot exceed 100,000 units")
    
    # Hedge ratio validation
    if not (0.0 <= hedge_ratio <= 1.0):
        errors.append("Hedge ratio must be between 0% and 100%")
    
    return errors


def render_multi_leg_strategy_selector():
    """Render multi-leg strategy selection interface."""
    st.markdown("**🎯 Multi-Leg Strategy Configuration**")
    
    strategy_type = st.selectbox(
        "Strategy Type:",
        options=[s.value for s in StrategyType],
        help="Select multi-leg options strategy",
        key="multi_leg_strategy_type"
    )
    
    try:
        current_price = get_current_price("WTI Crude Oil")
    except:
        current_price = 75.0
    
    # Get strategy defaults
    defaults = get_strategy_defaults(strategy_type, current_price)
    
    # Strategy-specific parameter inputs
    if strategy_type in ["Long Straddle", "Short Straddle"]:
        render_straddle_parameters(strategy_type, current_price, defaults)
    elif strategy_type in ["Long Strangle", "Short Strangle"]:
        render_strangle_parameters(strategy_type, current_price, defaults)
    elif strategy_type == "Collar":
        render_collar_parameters(current_price, defaults)
    elif strategy_type == "Butterfly Spread":
        render_butterfly_parameters(current_price, defaults)
    elif strategy_type == "Iron Condor":
        render_iron_condor_parameters(current_price, defaults)


def render_straddle_parameters(strategy_type, current_price, defaults):
    """Render straddle-specific parameters."""
    col1, col2 = st.columns(2)
    
    with col1:
        strike_price = st.slider(
            "Strike Price ($):",
            min_value=float(current_price * 0.8),
            max_value=float(current_price * 1.2),
            value=float(current_price),
            step=0.5,
            help="Both call and put strike",
            key="straddle_strike"
        )
        
        moneyness = current_price / strike_price
        if abs(moneyness - 1.0) < 0.02:
            st.caption("🎯 At-the-Money Straddle")
        elif moneyness > 1.02:
            st.caption("📉 Out-of-the-Money Straddle")
        else:
            st.caption("📈 In-the-Money Straddle")
    
    with col2:
        expiry_months = st.selectbox(
            "Expiry (months):",
            options=[1, 3, 6, 12],
            index=1,
            help="Time to expiration",
            key="straddle_expiry"
        )
        
        estimated_cost = estimate_strategy_cost(strategy_type, current_price, strike_price)
        st.metric("Est. Premium", f"${estimated_cost:.2f}")
    
    # Store configuration
    st.session_state.multi_leg_config = {
        'strategy_type': strategy_type,
        'strike_price': strike_price,
        'expiry_months': expiry_months
    }


def render_strangle_parameters(strategy_type, current_price, defaults):
    """Render strangle-specific parameters."""
    col1, col2 = st.columns(2)
    
    with col1:
        put_strike = st.slider(
            "Put Strike ($):",
            min_value=float(current_price * 0.7),
            max_value=float(current_price * 0.98),
            value=float(defaults.get('strikes', {}).get('put_strike', current_price * 0.95)),
            step=0.5,
            help="Put option strike price",
            key="strangle_put_strike"
        )
    
    with col2:
        call_strike = st.slider(
            "Call Strike ($):",
            min_value=float(current_price * 1.02),
            max_value=float(current_price * 1.3),
            value=float(defaults.get('strikes', {}).get('call_strike', current_price * 1.05)),
            step=0.5,
            help="Call option strike price",
            key="strangle_call_strike"
        )
    
    width = call_strike - put_strike
    st.caption(f"💰 Strike width: ${width:.2f}")
    
    expiry_months = st.selectbox(
        "Expiry (months):",
        options=[1, 3, 6, 12],
        index=1,
        key="strangle_expiry"
    )
    
    st.session_state.multi_leg_config = {
        'strategy_type': strategy_type,
        'call_strike': call_strike,
        'put_strike': put_strike,
        'expiry_months': expiry_months
    }


def render_collar_parameters(current_price, defaults):
    """Render collar-specific parameters."""
    col1, col2 = st.columns(2)
    
    with col1:
        put_strike = st.slider(
            "Put Strike ($) - Protection:",
            min_value=float(current_price * 0.7),
            max_value=float(current_price * 0.98),
            value=float(defaults.get('strikes', {}).get('put_strike', current_price * 0.90)),
            step=0.5,
            help="Downside protection level",
            key="collar_put_strike"
        )
        
        protection = (current_price - put_strike) / current_price * 100
        st.caption(f"🛡️ Protection: {protection:.1f}% downside")
    
    with col2:
        call_strike = st.slider(
            "Call Strike ($) - Cap:",
            min_value=float(current_price * 1.02),
            max_value=float(current_price * 1.3),
            value=float(defaults.get('strikes', {}).get('call_strike', current_price * 1.10)),
            step=0.5,
            help="Upside participation cap",
            key="collar_call_strike"
        )
        
        upside = (call_strike - current_price) / current_price * 100
        st.caption(f"📈 Upside cap: {upside:.1f}%")
    
    st.session_state.multi_leg_config = {
        'strategy_type': 'Collar',
        'call_strike': call_strike,
        'put_strike': put_strike,
        'expiry_months': 3
    }


def debug_delta_calculation():
    """Simple delta debug for Streamlit"""
    st.markdown("### 🔍 Delta Debug")
    
    if st.button("Check My Delta Calculation"):
        portfolio = st.session_state.portfolio_manager
        
        st.write("**Your Positions:**")
        total_delta = 0
        
        for name, position in portfolio.positions.items():
            # Get the delta for this position
            greeks = position.get_position_greeks()
            delta = greeks['delta']
            total_delta += delta
            
            # Show the details
            st.write(f"**{name}:**")
            st.write(f"- Strategy: {position.strategy}")
            st.write(f"- Size: {position.size:,.0f}")
            st.write(f"- Hedge Ratio: {position.hedge_ratio:.1%}")
            st.write(f"- Delta: {delta:.3f}")
            
            # If it's futures, show the calculation
            if position.strategy == "Futures":
                direction = 1 if position.size > 0 else -1
                calculation = direction * abs(position.size) * position.hedge_ratio
                st.write(f"- Calculation: {direction} × {abs(position.size):,.0f} × {position.hedge_ratio:.1f} = {calculation:.1f}")
            
            st.write("---")
        
        st.write(f"**Total Portfolio Delta: {total_delta:.3f}**")
        
        # Tell user if this looks wrong
        if abs(total_delta) > 1000:
            st.error("🚨 This delta is WAY too high! Something is wrong.")
        elif abs(total_delta) > 100:
            st.warning("⚠️ This delta seems high. Let's check if it's correct.")
        else:
            st.success("✅ This delta looks reasonable.")

def render_pnl_alerts(attribution_engine):
    """Render P&L alerts for risk management"""
    
    try:
        attribution_results = attribution_engine.calculate_daily_pnl_attribution()
        portfolio_pnl = attribution_results['portfolio_total']
        
        total_pnl = portfolio_pnl.get('total_pnl', 0)
        delta_pnl = portfolio_pnl.get('delta_pnl', 0)
        theta_pnl = portfolio_pnl.get('theta_pnl', 0)
        
        alerts = []
        
        if abs(total_pnl) > 5000:
            alert_type = "success" if total_pnl > 0 else "error"
            alerts.append({
                'type': alert_type,
                'message': f"Large P&L move: ${total_pnl:,.0f}",
                'action': "Review position sizing"
            })
        
        if abs(delta_pnl) > 3000:
            alerts.append({
                'type': 'warning',
                'message': f"High delta P&L: ${delta_pnl:,.0f}",
                'action': "Consider delta hedging"
            })
        
        if theta_pnl < -100:
            alerts.append({
                'type': 'info',
                'message': f"High time decay: ${theta_pnl:,.0f}",
                'action': "Monitor options expiry"
            })
        
        for alert in alerts:
            if alert['type'] == 'success':
                st.success(f"🟢 {alert['message']} - {alert['action']}")
            elif alert['type'] == 'error':
                st.error(f"🔴 {alert['message']} - {alert['action']}")
            elif alert['type'] == 'warning':
                st.warning(f"🟡 {alert['message']} - {alert['action']}")
            else:
                st.info(f"ℹ️ {alert['message']} - {alert['action']}")
                
    except Exception as e:
        st.error(f"Error generating P&L alerts: {e}")


def render_live_pnl_tracker(portfolio_manager):
    """Render live P&L tracking component"""
    
    st.markdown("### 📊 Live P&L Tracker")
    
    auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    if st.button("🔄 Refresh P&L Now", type="secondary"):
        st.rerun()
    
    attribution_engine = PnLAttributionEngine(portfolio_manager)
    
    try:
        attribution_results = attribution_engine.calculate_daily_pnl_attribution()
        portfolio_pnl = attribution_results['portfolio_total']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_pnl = portfolio_pnl.get('total_pnl', 0)
            pnl_change = "+$" if total_pnl >= 0 else "-$"
            st.metric(
                "Live P&L", 
                f"${total_pnl:,.0f}",
                f"{pnl_change}{abs(total_pnl):,.0f} today"
            )
        
        with col2:
            delta_pnl = portfolio_pnl.get('delta_pnl', 0)
            st.metric(
                "Delta P&L",
                f"${delta_pnl:,.0f}",
                "Price moves"
            )
        
        with col3:
            theta_pnl = portfolio_pnl.get('theta_pnl', 0)
            st.metric(
                "Theta P&L",
                f"${theta_pnl:,.0f}",
                "Time decay"
            )
        
        # P&L alerts
        render_pnl_alerts(attribution_engine)
        
    except Exception as e:
        st.error(f"Error in live P&L tracking: {e}")

def clear_streamlit_cache():
    """Clear any cached content that might cause overlays"""
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()

# ==========================================================================
# MAIN APPLICATION ENTRY POINT
# ==========================================================================
def main():
    initialize_session_state()
    
    st.markdown("""
    <div class="hero-header-premium fade-in">
        <div class="hero-title-premium">Commodity Hedging Platform</div>
        <div class="hero-subtitle-premium">Institutional-grade multi-commodity risk management with real-time P&L attribution</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        📊 <strong>Portfolio Management</strong>: Multi-commodity portfolio analysis with real-time P&L attribution, 
        advanced risk metrics, multi-leg options strategies, and comprehensive Greeks monitoring
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    portfolio_interface()


if __name__ == "__main__":
    main()