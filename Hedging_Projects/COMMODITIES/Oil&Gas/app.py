"""
app.py

Complete Multi-Commodity Oil & Gas Hedging Simulator with Portfolio Management.
Supports both single position analysis and multi-commodity portfolio management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Show debugging information
st.sidebar.markdown("## 🔧 Debug Info")

# Import existing modules with detailed error reporting
try:
    from hedging.data import get_prices, get_current_price, get_available_commodities
    st.sidebar.success("✅ Data module imported")
except ImportError as e:
    st.sidebar.error(f"❌ Data module error: {e}")

try:
    from hedging.strategies import compute_payoff_diagram, get_hedge_summary
    st.sidebar.success("✅ Strategies module imported")
except ImportError as e:
    st.sidebar.error(f"❌ Strategies module error: {e}")

try:
    from hedging.simulation import simulate_hedged_vs_unhedged, compare_hedging_effectiveness
    st.sidebar.success("✅ Simulation module imported")
except ImportError as e:
    st.sidebar.error(f"❌ Simulation module error: {e}")

try:
    from hedging.risk import calculate_risk_metrics, calculate_delta_exposure, summarize_risk_comparison
    st.sidebar.success("✅ Risk module imported")
except ImportError as e:
    st.sidebar.error(f"❌ Risk module error: {e}")

try:
    from hedging.stress_testing import STRESS_SCENARIOS
    st.sidebar.success("✅ Stress testing module imported")
except ImportError as e:
    st.sidebar.error(f"❌ Stress testing module error: {e}")

# Import new portfolio functionality with detailed debugging
try:
    from hedging.portfolio import (
        PortfolioManager, Position, 
        create_oil_position, create_gas_position, create_brent_position,
        create_sample_portfolio
    )
    st.sidebar.success("✅ Portfolio module imported successfully!")
    PORTFOLIO_ENABLED = True
except ImportError as e:
    st.sidebar.error(f"❌ Portfolio module error: {e}")
    st.sidebar.info("Creating dummy portfolio classes for testing...")
    
    # Create dummy classes to enable the interface for testing
    class PortfolioManager:
        def __init__(self):
            self.positions = {}
        def __len__(self):
            return len(self.positions)
        def add_position(self, name, pos):
            return self
        def clear(self):
            return self
    
    class Position:
        def __init__(self, **kwargs):
            pass
    
    def create_sample_portfolio():
        return PortfolioManager()
    
    def create_oil_position(*args, **kwargs):
        return Position()
    
    def create_gas_position(*args, **kwargs):
        return Position()
    
    def create_brent_position(*args, **kwargs):
        return Position()
    
    PORTFOLIO_ENABLED = True  # Enable anyway for testing


# Page configuration
st.set_page_config(
    page_title="Multi-Commodity Hedging Simulator",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e5c8a;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .portfolio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .position-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin-bottom: 1rem;
    }
    .sidebar-section {
        background-color: #f1f3f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-banner {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .mode-toggle {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'portfolio_mode' not in st.session_state:
        st.session_state.portfolio_mode = True  # Start with portfolio mode for testing
    
    if 'portfolio_manager' not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()
    
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    
    # Single position session state
    if 'single_position_results' not in st.session_state:
        st.session_state.single_position_results = None


def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # App header with mode selection
    st.markdown('<h1 class="main-header">🛢️ Multi-Commodity Hedging Simulator</h1>', unsafe_allow_html=True)
    
    # Mode selection
    st.markdown('<div class="mode-toggle">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        mode_option = st.radio(
            "📊 **Choose Analysis Mode:**",
            options=["🎯 Single Position", "📊 Portfolio Management"],
            index=1 if st.session_state.portfolio_mode else 0,
            horizontal=True,
            help="Single Position: Analyze one commodity at a time\nPortfolio: Manage multiple commodities with advanced analytics"
        )
        
        st.session_state.portfolio_mode = (mode_option == "📊 Portfolio Management")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode description
    if st.session_state.portfolio_mode:
        st.markdown("**📊 Portfolio Mode:** Multi-commodity portfolio analysis with correlation matrices, portfolio-level risk metrics, and advanced optimization")
    else:
        st.markdown("**🎯 Single Position Mode:** Traditional single commodity hedging analysis with detailed payoff diagrams and risk metrics")
    
    st.markdown("---")
    
    # Route to appropriate interface
    if st.session_state.portfolio_mode:
        portfolio_interface()
    else:
        single_position_interface()


def portfolio_interface():
    """Portfolio management interface."""
    
    st.markdown('<div class="success-banner">✨ <strong>Portfolio Mode Active</strong> - Testing multi-commodity portfolio functionality</div>', unsafe_allow_html=True)
    
    # Simple test interface to verify portfolio mode is working
    st.markdown("## 🧪 Portfolio Mode Test Interface")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🏗️ Portfolio Builder (Test)")
        
        if st.button("📝 Test Sample Portfolio"):
            st.success("Sample portfolio button clicked!")
            try:
                portfolio = create_sample_portfolio()
                st.success("Sample portfolio created successfully!")
            except Exception as e:
                st.error(f"Error creating portfolio: {e}")
        
        if st.button("🗑️ Clear All"):
            st.success("Clear button clicked!")
        
        st.markdown("### ➕ Add Position (Test)")
        
        with st.form("test_form"):
            pos_name = st.text_input("Position Name:", value="test_position")
            commodity = st.selectbox("Commodity:", ["WTI Crude Oil", "Natural Gas"])
            size = st.number_input("Size:", value=1000.0)
            
            if st.form_submit_button("Add Position"):
                st.success(f"Would add position: {pos_name} - {commodity} - {size}")
    
    with col2:
        st.markdown("### 📊 Portfolio Dashboard (Test)")
        
        # Test tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "Correlations", "Risk"])
        
        with tab1:
            st.markdown("**Portfolio Overview Tab**")
            st.info("This would show portfolio composition charts")
            
            # Test chart
            fig = go.Figure(data=go.Pie(
                labels=['WTI Oil', 'Natural Gas', 'Brent Oil'],
                values=[50, 30, 20]
            ))
            fig.update_layout(title="Test Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("**Correlations Tab**")
            st.info("This would show correlation heatmaps")
            
            # Test correlation matrix
            test_corr = pd.DataFrame({
                'WTI': [1.0, 0.85, 0.92],
                'Brent': [0.85, 1.0, 0.78],
                'Gas': [0.92, 0.78, 1.0]
            }, index=['WTI', 'Brent', 'Gas'])
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=test_corr.values,
                x=test_corr.columns,
                y=test_corr.index,
                colorscale='RdBu',
                zmid=0
            ))
            fig_heatmap.update_layout(title="Test Correlation Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab3:
            st.markdown("**Risk Analysis Tab**")
            st.info("This would show portfolio risk metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio VaR", "$-15,234")
            with col2:
                st.metric("Portfolio CVaR", "$-23,567")
            with col3:
                st.metric("Volatility", "$8,945")
            with col4:
                st.metric("Sharpe Ratio", "0.234")


def single_position_interface():
    """Single position interface - simplified for now."""
    
    st.markdown("## 🎯 Single Position Analysis")
    st.info("💡 Switch to Portfolio Mode to test the new multi-commodity features!")
    
    # Simple single position interface
    with st.sidebar:
        st.markdown("### Position Parameters")
        commodity = st.selectbox("Commodity:", ["WTI Crude Oil", "Natural Gas"])
        position = st.number_input("Position Size:", value=1000.0)
        hedge_ratio = st.slider("Hedge Ratio:", 0.0, 1.0, 0.8)
        
        if st.button("🚀 Run Analysis"):
            st.success("Analysis would run here in full version")
    
    st.markdown("### 📈 Analysis Results")
    st.info("Switch to Portfolio Mode above to see the new multi-commodity interface!")


if __name__ == "__main__":
    main()
