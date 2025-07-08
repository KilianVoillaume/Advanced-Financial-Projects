"""
app.py

Professional Multi-Commodity Oil & Gas Hedging Simulator.
Beautiful UI with portfolio management and advanced analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import all modules (we know they work now!)
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
    page_title="Multi-Commodity Hedging Platform",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styles */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Mode Toggle */
    .mode-selector {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .mode-title {
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Cards */
    .portfolio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .portfolio-card:hover {
        transform: translateY(-5px);
    }
    
    .position-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .position-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-top: 4px solid #4ECDC4;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Sidebar */
    .sidebar-section {
        background: linear-gradient(145deg, #f8f9ff 0%, #e8f4fd 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .sidebar-title {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Success Banner */
    .success-banner {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.3);
        font-weight: 500;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f7fafc;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'portfolio_mode' not in st.session_state:
        st.session_state.portfolio_mode = True  # Default to portfolio mode
    
    if 'portfolio_manager' not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()
    
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    
    if 'single_position_results' not in st.session_state:
        st.session_state.single_position_results = None


def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Hero Header
    st.markdown("""
    <div class="hero-header fade-in">
        <div class="hero-title">🛢️ Oil & Gas Hedging Platform</div>
        <div class="hero-subtitle">Professional multi-commodity risk management and portfolio optimization</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selection
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    st.markdown('<div class="mode-title">Choose Your Analysis Mode</div>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode Description
    if st.session_state.portfolio_mode:
        st.info("📊 **Portfolio Mode**: Multi-commodity portfolio analysis with correlation matrices, advanced risk metrics, and optimization algorithms")
    else:
        st.info("🎯 **Single Position Mode**: Deep-dive analysis of individual commodity positions with detailed payoff diagrams and Monte Carlo simulations")
    
    st.markdown("---")
    
    # Route to appropriate interface
    if st.session_state.portfolio_mode:
        portfolio_interface()
    else:
        single_position_interface()


def portfolio_interface():
    """Professional portfolio management interface."""
    
    st.markdown("""
    <div class="success-banner fade-in">
        ✨ <strong>Portfolio Mode Active</strong> - Advanced multi-commodity risk management platform
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        portfolio_builder_sidebar()
    
    with col2:
        portfolio_dashboard()


def portfolio_builder_sidebar():
    """Professional portfolio builder sidebar."""
    
    st.markdown('<div class="section-header">🏗️ Portfolio Builder</div>', unsafe_allow_html=True)
    
    # Portfolio summary card
    portfolio = st.session_state.portfolio_manager
    
    if len(portfolio) > 0:
        total_notional = sum(pos.notional_value for pos in portfolio.positions.values())
        unique_commodities = len(set(pos.commodity for pos in portfolio.positions.values()))
        
        st.markdown(f"""
        <div class="portfolio-card fade-in">
            <h3 style="margin-bottom: 1rem;">📊 Portfolio Overview</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <div style="font-size: 2rem; font-weight: bold;">{len(portfolio)}</div>
                    <div style="opacity: 0.9;">Positions</div>
                </div>
                <div>
                    <div style="font-size: 2rem; font-weight: bold;">{unique_commodities}</div>
                    <div style="opacity: 0.9;">Commodities</div>
                </div>
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
                <div style="font-size: 1.5rem; font-weight: bold;">${total_notional:,.0f}</div>
                <div style="opacity: 0.9;">Total Notional Value</div>
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
            st.success("Sample portfolio loaded!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            st.session_state.portfolio_manager.clear()
            st.success("Portfolio cleared!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add new position
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
            position_size = st.number_input(
                "Size:",
                min_value=-100000.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0,
                help="Positive=Long, Negative=Short"
            )
        
        with col2:
            hedge_ratio = st.slider(
                "Hedge %:",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                format="%.0f%%",
                help="Percentage to hedge"
            )
        
        strategy = st.selectbox(
            "Strategy:",
            options=["Futures", "Options"],
            help="Hedging instrument"
        )
        
        strike_price = None
        if strategy == "Options":
            try:
                current_price = get_current_price(commodity)
                strike_price = st.slider(
                    "Strike Price:",
                    min_value=float(current_price * 0.8),
                    max_value=float(current_price * 1.2),
                    value=float(current_price),
                    step=0.5
                )
                
                # Moneyness indicator
                moneyness = current_price / strike_price
                if abs(moneyness - 1.0) < 0.05:
                    st.caption("🎯 At-the-Money (ATM)")
                elif moneyness > 1.05:
                    st.caption("📉 Out-of-the-Money (OTM)")
                else:
                    st.caption("📈 In-the-Money (ITM)")
                    
            except:
                strike_price = st.number_input("Strike Price:", value=75.0)
        
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
                st.error(f"Position '{position_name}' already exists!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current positions
    if len(portfolio) > 0:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📋 Current Positions</div>', unsafe_allow_html=True)
        
        for name, position in portfolio.positions.items():
            direction_emoji = "📈" if position.size > 0 else "📉"
            hedge_status = "🛡️" if position.hedge_ratio > 0.5 else "⚠️" if position.hedge_ratio > 0 else "🚫"
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="position-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{direction_emoji} {name}</strong><br>
                                <small style="color: #666;">
                                    {position.commodity} • {abs(position.size):,.0f} • {position.hedge_ratio:.0%} hedged {hedge_status}
                                </small>
                            </div>
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
    
    # Update config
    st.session_state.portfolio_manager.set_config(
        confidence_level=confidence_level,
        simulation_runs=n_simulations
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    st.markdown("---")
    
    if st.button("🚀 Run Portfolio Analysis", type="primary", use_container_width=True, disabled=len(portfolio) == 0):
        if len(portfolio) > 0:
            st.session_state.simulation_run = True
            st.rerun()


def portfolio_dashboard():
    """Professional portfolio dashboard."""
    
    portfolio = st.session_state.portfolio_manager
    
    if len(portfolio) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 15px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h2 style="color: #2d3748; margin-bottom: 1rem;">Ready to Build Your Portfolio?</h2>
            <p style="color: #718096; font-size: 1.1rem; margin-bottom: 1.5rem;">
                Start by adding commodity positions using the builder on the left
            </p>
            <div style="background: white; padding: 1rem; border-radius: 8px; display: inline-block;">
                💡 <strong>Tip:</strong> Try the "Load Sample" button for a quick start!
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Portfolio analytics
    with st.spinner("🔄 Calculating portfolio analytics..."):
        try:
            portfolio.calculate_correlations().calculate_portfolio_risk()
            analysis_ready = True
        except Exception as e:
            st.error(f"Error in analytics: {e}")
            analysis_ready = False
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🔗 Correlations", 
        "⚠️ Risk Metrics", 
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
    """Portfolio overview tab with beautiful visualizations."""
    
    st.markdown('<div class="section-header">📊 Portfolio Composition</div>', unsafe_allow_html=True)
    
    # Summary metrics
    if analysis_ready:
        risk_summary = portfolio.get_portfolio_risk_summary()
        if risk_summary:
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = list(risk_summary.items())
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #4ECDC4;">Expected P&L</h4>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2d3748;">{metrics[0][1]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #FF6B6B;">VaR (95%)</h4>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2d3748;">{metrics[1][1]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #667eea;">Volatility</h4>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2d3748;">{metrics[3][1]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #48bb78;">Sharpe Ratio</h4>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2d3748;">{metrics[4][1]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Portfolio composition
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Portfolio allocation chart
        weights = portfolio.get_portfolio_weights()
        if weights:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.5,
                textinfo='label+percent',
                textposition='auto',
                marker=dict(
                    colors=['#667eea', '#764ba2', '#4ECDC4', '#FF6B6B', '#48bb78'],
                    line=dict(color='white', width=2)
                )
            )])
            
            fig_pie.update_layout(
                title={
                    'text': "Portfolio Allocation by Position",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'family': 'Inter'}
                },
                showlegend=True,
                height=400,
                margin=dict(t=60, b=20, l=20, r=20),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.05
                )
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Portfolio summary table
        summary_df = portfolio.get_portfolio_summary()
        if not summary_df.empty:
            st.markdown("**Position Details:**")
            
            # Simplified summary for better display
            display_df = summary_df[['Position Name', 'Commodity', 'Direction', 'Weight', 'Hedge Ratio']].copy()
            st.dataframe(
                display_df, 
                use_container_width=True, 
                hide_index=True,
                height=350
            )
        
        # Commodity exposure
        exposure_df = portfolio.get_commodity_exposure()
        if not exposure_df.empty:
            st.markdown("**Net Commodity Exposure:**")
            st.dataframe(exposure_df, use_container_width=True, hide_index=True)


def correlations_tab(portfolio, analysis_ready):
    """Correlations tab with beautiful heatmap."""
    
    st.markdown('<div class="section-header">🔗 Cross-Commodity Correlations</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to calculate correlations")
        return
    
    corr_matrix = portfolio.get_correlation_matrix()
    
    if corr_matrix.empty:
        st.info("ℹ️ Add positions with different commodities to see correlations")
        return
    
    # Beautiful correlation heatmap
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
        textfont={"size": 14, "color": "white"},
        colorbar=dict(title="Correlation")
    ))
    
    fig_heatmap.update_layout(
        title={
            'text': "Commodity Correlation Matrix",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter'}
        },
        height=500,
        xaxis_title="Commodities",
        yaxis_title="Commodities",
        font=dict(family="Inter", size=12)
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Correlation insights
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
        st.dataframe(corr_df, use_container_width=True, hide_index=True)


def risk_analysis_tab(portfolio, analysis_ready):
    """Risk analysis tab with comprehensive metrics."""
    
    st.markdown('<div class="section-header">⚠️ Portfolio Risk Analysis</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to see risk metrics")
        return
    
    # P&L Distribution
    try:
        portfolio_pnl = portfolio._simulate_portfolio_pnl()
        
        if len(portfolio_pnl) > 0:
            # Beautiful P&L distribution chart
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=portfolio_pnl,
                nbinsx=50,
                name='Portfolio P&L',
                marker=dict(
                    color='rgba(102, 126, 234, 0.7)',
                    line=dict(color='rgba(102, 126, 234, 1)', width=1)
                ),
                opacity=0.8
            ))
            
            # Add VaR and CVaR lines
            var_95 = np.percentile(portfolio_pnl, 5)
            cvar_95 = np.mean(portfolio_pnl[portfolio_pnl <= var_95])
            
            fig_dist.add_vline(
                x=var_95,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"VaR (95%): ${var_95:,.0f}"
            )
            
            fig_dist.add_vline(
                x=cvar_95,
                line_dash="dot",
                line_color="darkred",
                line_width=3,
                annotation_text=f"CVaR (95%): ${cvar_95:,.0f}"
            )
            
            fig_dist.update_layout(
                title={
                    'text': "Portfolio P&L Distribution (Monte Carlo Simulation)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'family': 'Inter'}
                },
                xaxis_title="P&L ($)",
                yaxis_title="Frequency",
                height=400,
                showlegend=False,
                font=dict(family="Inter", size=12)
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Risk metrics summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #FF6B6B;">Worst Case Loss</h4>
                    <div style="font-size: 1.8rem; font-weight: bold;">${var_95:,.0f}</div>
                    <small style="color: #666;">Value at Risk (95%)</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #dc3545;">Tail Risk</h4>
                    <div style="font-size: 1.8rem; font-weight: bold;">${cvar_95:,.0f}</div>
                    <small style="color: #666;">Conditional VaR (95%)</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                prob_loss = np.sum(portfolio_pnl < 0) / len(portfolio_pnl) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea;">Loss Probability</h4>
                    <div style="font-size: 1.8rem; font-weight: bold;">{prob_loss:.1f}%</div>
                    <small style="color: #666;">Chance of losing money</small>
                </div>
                """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Could not generate risk analysis: {e}")


def stress_testing_tab(portfolio, analysis_ready):
    """Stress testing tab with historical scenarios."""
    
    st.markdown('<div class="section-header">🧪 Historical Crisis Stress Testing</div>', unsafe_allow_html=True)
    
    if not analysis_ready:
        st.warning("⚠️ Run portfolio analysis to enable stress testing")
        return
    
    st.markdown("""
    Test your portfolio against major historical market crises to understand 
    how it would have performed during extreme market conditions.
    """)
    
    # Scenario selection
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
        # Create stress test results
        stress_results = []
        
        for scenario_name in selected_scenarios:
            scenario = STRESS_SCENARIOS[scenario_name]
            
            # Calculate portfolio impact
            total_unhedged = 0
            total_hedged = 0
            
            for name, position in portfolio.positions.items():
                # Get commodity type
                commodity_type = "gas" if "gas" in position.commodity.lower() else "oil"
                
                # Calculate price shock
                price_shock = scenario["oil_peak_to_trough"]
                if commodity_type == "gas":
                    price_shock *= scenario["gas_correlation"]
                
                # Position P&L
                price_change = position.current_price * price_shock
                unhedged_pnl = price_change * position.size
                
                # Hedge P&L
                if position.strategy == "Futures":
                    hedge_pnl = -price_change * position.size * position.hedge_ratio
                elif position.strategy == "Options":
                    if price_change < 0:  # Downside protection
                        hedge_pnl = -price_change * position.size * position.hedge_ratio * 0.8
                    else:  # Limited upside protection
                        hedge_pnl = -price_change * position.size * position.hedge_ratio * 0.2
                else:
                    hedge_pnl = 0
                
                total_unhedged += unhedged_pnl
                total_hedged += unhedged_pnl + hedge_pnl
            
            # Add to results
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
        
        # Display results table
        st.markdown("### 📊 Stress Test Results")
        results_df = pd.DataFrame(stress_results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Key insights
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
                    <h4 style="color: #FF6B6B;">Worst Unhedged Loss</h4>
                    <div style="font-size: 1.5rem; font-weight: bold;">${worst_unhedged:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #4ECDC4;">Worst Hedged Loss</h4>
                    <div style="font-size: 1.5rem; font-weight: bold;">${worst_hedged:,.0f}</div>
                    <small style="color: #28a745;">Improvement: ${worst_hedged - worst_unhedged:,.0f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_reduction = (abs(worst_unhedged) - abs(worst_hedged)) / abs(worst_unhedged) * 100 if worst_unhedged != 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #48bb78;">Risk Reduction</h4>
                    <div style="font-size: 1.5rem; font-weight: bold;">{risk_reduction:.1f}%</div>
                    <small style="color: #666;">Average protection: {avg_protection:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Select crisis scenarios above to run stress tests")


def single_position_interface():
    """Single position interface for traditional analysis."""
    
    st.markdown('<div class="section-header">🎯 Single Position Analysis</div>', unsafe_allow_html=True)
    
    # Coming soon message with beautiful styling
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 15px; margin: 2rem 0;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🚧</div>
        <h2 style="color: #2d3748; margin-bottom: 1rem;">Single Position Mode</h2>
        <p style="color: #718096; font-size: 1.1rem; margin-bottom: 1.5rem;">
            Traditional single commodity analysis is being enhanced with the new design
        </p>
        <div style="background: white; padding: 1.5rem; border-radius: 12px; display: inline-block; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
            <strong>💡 Meanwhile, explore the powerful Portfolio Mode above!</strong><br>
            <span style="color: #666;">Manage multiple commodities with advanced risk analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Helper functions
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


if __name__ == "__main__":
    main()
