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

# Import existing modules
try:
    from hedging.data import get_prices, get_current_price, get_available_commodities
    from hedging.strategies import compute_payoff_diagram, get_hedge_summary
    from hedging.simulation import simulate_hedged_vs_unhedged, compare_hedging_effectiveness
    from hedging.risk import calculate_risk_metrics, calculate_delta_exposure, summarize_risk_comparison
    from hedging.stress_testing import STRESS_SCENARIOS
    
    # Import new portfolio functionality
    from hedging.portfolio import (
        PortfolioManager, Position, 
        create_oil_position, create_gas_position, create_brent_position,
        create_sample_portfolio
    )
    PORTFOLIO_ENABLED = True
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please ensure all hedging modules are properly installed.")
    PORTFOLIO_ENABLED = False


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
        st.session_state.portfolio_mode = False  # Start with single position
    
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
    st.markdown('<h1 class="main-header">🛢️ Oil & Gas Hedging Simulator</h1>', unsafe_allow_html=True)
    
    # Mode selection
    if PORTFOLIO_ENABLED:
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
    
    else:
        st.error("⚠️ Portfolio functionality unavailable. Running in Single Position mode only.")
        st.session_state.portfolio_mode = False
    
    st.markdown("---")
    
    # Route to appropriate interface
    if st.session_state.portfolio_mode and PORTFOLIO_ENABLED:
        portfolio_interface()
    else:
        single_position_interface()


def single_position_interface():
    """Complete single position analysis interface (your original functionality)."""
    
    st.markdown("## 🎯 Single Position Hedging Analysis")
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("## 📊 Simulation Parameters")
        
        # Commodity selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🏭 Commodity Selection")
        
        commodity = st.selectbox(
            "Select Commodity:",
            options=["WTI Crude Oil", "Brent Crude Oil", "Natural Gas"],
            index=0,
            help="Select the commodity for hedging analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Position parameters
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📈 Position Parameters")
        
        position = st.number_input(
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
        
        # Adjust position sign based on type
        if position_type == "Short":
            position = -position
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Hedging strategy parameters
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🛡️ Hedging Strategy")
        
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
            format="%.2f",
            help="Percentage of position to hedge (0.0 = no hedge, 1.0 = full hedge)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Options-specific parameters
        strike_price = None
        option_expiry = None
        
        if strategy == "Options":
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Options Parameters")
            
            try:
                current_price = get_current_price(commodity)
                
                strike_price = st.slider(
                    "Strike Price ($):",
                    min_value=float(current_price * 0.7),
                    max_value=float(current_price * 1.3),
                    value=float(current_price),
                    step=0.5,
                    help="Option strike price (default is at-the-money)"
                )
                
                # Moneyness indicator
                moneyness = float(current_price) / float(strike_price)
                if abs(moneyness - 1.0) < 0.05:
                    moneyness_desc = "At-the-Money (ATM)"
                elif moneyness > 1.05:
                    moneyness_desc = "Out-of-the-Money (OTM)"
                else:
                    moneyness_desc = "In-the-Money (ITM)"
                
                st.caption(f"Current Price: ${float(current_price):.2f} | Moneyness: {moneyness_desc}")
                
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
        
        # Risk analysis parameters
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Risk Analysis")
        
        confidence = st.slider(
            "Confidence Level:",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            format="%d%%",
            help="Confidence level for VaR and CVaR calculations"
        ) / 100.0
        
        n_simulations = st.selectbox(
            "Number of Simulations:",
            options=[1000, 5000, 10000],
            index=1,
            help="More simulations = more accurate results but slower computation"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Run simulation button
        st.markdown("---")
        run_simulation = st.button(
            "🚀 Run Simulation",
            type="primary",
            help="Click to run the hedging simulation"
        )
    
    # Main content area
    if run_simulation:
        st.session_state.simulation_run = True
    
        with st.spinner("Loading price data and running simulation..."):
            try:
                # Fetch data
                prices = get_prices(commodity)
                current_price = float(get_current_price(commodity))
                
                # Ensure strike_price is float for options
                if strategy == "Options" and strike_price is not None:
                    strike_price = float(strike_price)
                
                # Run simulation
                sim_results = simulate_hedged_vs_unhedged(
                    prices, position, hedge_ratio, strategy, strike_price, n_simulations
                )
                
                # Calculate payoff diagram
                payoff_data = compute_payoff_diagram(
                    float(current_price), position, hedge_ratio, strategy, 
                    float(strike_price) if strike_price is not None else None
                )
                
                # Calculate risk metrics
                hedged_risk = calculate_risk_metrics(sim_results['hedged_pnl'], confidence)
                unhedged_risk = calculate_risk_metrics(sim_results['unhedged_pnl'], confidence)
                
                # Calculate delta exposure
                delta_exposure = calculate_delta_exposure(
                    prices, position, hedge_ratio, strategy, 
                    float(strike_price) if strike_price is not None else None
                )
                
                # Store results in session state
                st.session_state.single_position_results = {
                    'prices': prices,
                    'current_price': current_price,
                    'sim_results': sim_results,
                    'payoff_data': payoff_data,
                    'hedged_risk': hedged_risk,
                    'unhedged_risk': unhedged_risk,
                    'delta_exposure': delta_exposure,
                    'params': {
                        'commodity': commodity,
                        'position': position,
                        'strategy': strategy,
                        'hedge_ratio': hedge_ratio,
                        'strike_price': strike_price,
                        'confidence': confidence
                    }
                }
                
                st.success("✅ Simulation completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error running simulation: {str(e)}")
                st.error("**Debug Information:**")
                st.error(f"- Commodity: {commodity}")
                st.error(f"- Strategy: {strategy}")
                st.error(f"- Position: {position}")
                st.error(f"- Hedge Ratio: {hedge_ratio}")
                if strategy == "Options":
                    st.error(f"- Strike Price: {strike_price}")
                
                st.info("💡 **Suggestions:**")
                st.info("- Try a different commodity (WTI usually works best)")
                st.info("- Check your internet connection")
                st.info("- Try with a smaller position size")
                st.info("- Switch to Futures strategy if Options is failing")
                
                st.session_state.simulation_run = False
    
    # Display results if simulation has been run
    if st.session_state.simulation_run and st.session_state.single_position_results:
        display_single_position_results()


def display_single_position_results():
    """Display single position simulation results."""
    
    results = st.session_state.single_position_results
    params = results['params']
    
    # Summary metrics at the top
    st.markdown('<h2 class="section-header">📋 Simulation Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Commodity",
            params['commodity'],
            help="Selected commodity for analysis"
        )
    
    with col2:
        st.metric(
            "Current Price",
            f"${results['current_price']:.2f}",
            help="Most recent market price"
        )
    
    with col3:
        st.metric(
            "Position Size",
            f"{abs(params['position']):,.0f}",
            delta=f"{'Long' if params['position'] > 0 else 'Short'} Position",
            help="Position size and direction"
        )
    
    with col4:
        st.metric(
            "Hedge Ratio",
            f"{params['hedge_ratio']:.1%}",
            delta=f"{params['strategy']} Strategy",
            help="Percentage of position hedged"
        )
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Chart", 
        "📊 Payoff Diagram", 
        "🎯 P&L Distribution", 
        "📋 Risk Metrics", 
        "⚠️ Stress Testing"
    ])
    
    with tab1:
        display_price_chart(results)
    
    with tab2:
        display_payoff_diagram(results)
    
    with tab3:
        display_pnl_distribution(results)
    
    with tab4:
        display_risk_metrics(results)
    
    with tab5:
        display_stress_testing(results)


def display_price_chart(results):
    """Display historical price chart."""
    
    st.markdown('<h3 class="section-header">Historical Price Chart</h3>', unsafe_allow_html=True)
    
    prices = results['prices']
    commodity = results['params']['commodity']
    current_price = results['current_price']
    
    # Create interactive price chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        mode='lines',
        name=f'{commodity} Price',
        line=dict(color='#1f4e79', width=2)
    ))
    
    # Add current price line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    # Add strike price line for options
    if results['params']['strategy'] == 'Options' and results['params']['strike_price']:
        strike_price = results['params']['strike_price']
        fig.add_hline(
            y=strike_price,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Strike: ${strike_price:.2f}"
        )
    
    fig.update_layout(
        title=f"{commodity} Price History",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Price statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_price = float(prices.min())
        st.metric("Min Price", f"${min_price:.2f}")
    
    with col2:
        max_price = float(prices.max())
        st.metric("Max Price", f"${max_price:.2f}")
    
    with col3:
        returns = prices.pct_change().dropna()
        if len(returns) > 0:
            volatility = float(returns.std()) * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.1f}%")
        else:
            st.metric("Annualized Volatility", "N/A")


def display_payoff_diagram(results):
    """Display payoff diagram."""
    
    st.markdown('<h3 class="section-header">Payoff Diagram at Expiry</h3>', unsafe_allow_html=True)
    st.markdown("This diagram shows profit/loss for different price scenarios at expiration.")
    
    payoff_data = results['payoff_data']
    current_price = results['current_price']
    
    # Create payoff diagram
    fig = go.Figure()
    
    # Add underlying position P&L
    fig.add_trace(go.Scatter(
        x=payoff_data['spot_prices'],
        y=payoff_data['underlying_pnl'],
        mode='lines',
        name='Underlying P&L',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add hedge P&L
    fig.add_trace(go.Scatter(
        x=payoff_data['spot_prices'],
        y=payoff_data['hedge_pnl'],
        mode='lines',
        name='Hedge P&L',
        line=dict(color='blue', width=2, dash='dot')
    ))
    
    # Add net P&L
    fig.add_trace(go.Scatter(
        x=payoff_data['spot_prices'],
        y=payoff_data['net_pnl'],
        mode='lines',
        name='Net P&L',
        line=dict(color='green', width=3)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    # Add current price line
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Current Price: ${current_price:.2f}"
    )
    
    # Add breakeven points
    for breakeven in payoff_data['breakeven_prices']:
        fig.add_vline(
            x=breakeven,
            line_dash="dot",
            line_color="purple",
            annotation_text=f"Breakeven: ${breakeven:.2f}"
        )
    
    fig.update_layout(
        title="Payoff Diagram: Hedged vs Unhedged Position",
        xaxis_title="Spot Price at Expiry ($)",
        yaxis_title="Profit & Loss ($)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Payoff summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_profit = float(payoff_data['net_pnl'].max())
        st.metric("Max Profit", f"${max_profit:,.0f}")
    
    with col2:
        max_loss = float(payoff_data['net_pnl'].min())
        st.metric("Max Loss", f"${max_loss:,.0f}")
    
    with col3:
        num_breakevens = len(payoff_data['breakeven_prices'])
        st.metric("Breakeven Points", f"{num_breakevens}")


def display_pnl_distribution(results):
    """Display P&L distribution histogram."""
    
    st.markdown('<h3 class="section-header">P&L Distribution Analysis</h3>', unsafe_allow_html=True)
    st.markdown("Monte Carlo simulation results showing probability distribution of outcomes.")
    
    sim_results = results['sim_results']
    
    # Create subplot with two histograms
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Unhedged Position', 'Hedged Position'),
        shared_yaxes=True
    )
    
    # Unhedged histogram
    fig.add_trace(
        go.Histogram(
            x=sim_results['unhedged_pnl'],
            name='Unhedged P&L',
            opacity=0.7,
            marker_color='red',
            nbinsx=50
        ),
        row=1, col=1
    )
    
    # Hedged histogram
    fig.add_trace(
        go.Histogram(
            x=sim_results['hedged_pnl'],
            name='Hedged P&L',
            opacity=0.7,
            marker_color='blue',
            nbinsx=50
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="P&L Distribution Comparison",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="P&L ($)")
    fig.update_yaxes(title_text="Frequency")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution statistics
    st.markdown("#### Distribution Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Unhedged Position:**")
        unhedged_mean = float(np.mean(sim_results['unhedged_pnl']))
        unhedged_std = float(np.std(sim_results['unhedged_pnl']))
        unhedged_prob_loss = float(np.sum(sim_results['unhedged_pnl'] < 0) / len(sim_results['unhedged_pnl']))
        
        st.metric("Expected P&L", f"${unhedged_mean:,.0f}")
        st.metric("Volatility", f"${unhedged_std:,.0f}")
        st.metric("Probability of Loss", f"{unhedged_prob_loss:.1%}")
    
    with col2:
        st.markdown("**Hedged Position:**")
        hedged_mean = float(np.mean(sim_results['hedged_pnl']))
        hedged_std = float(np.std(sim_results['hedged_pnl']))
        hedged_prob_loss = float(np.sum(sim_results['hedged_pnl'] < 0) / len(sim_results['hedged_pnl']))
        
        st.metric("Expected P&L", f"${hedged_mean:,.0f}")
        st.metric("Volatility", f"${hedged_std:,.0f}")
        st.metric("Probability of Loss", f"{hedged_prob_loss:.1%}")
    
    # Hedging effectiveness
    effectiveness = compare_hedging_effectiveness(sim_results['hedged_pnl'], sim_results['unhedged_pnl'])
    
    st.markdown("#### Hedging Effectiveness")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Volatility Reduction",
            f"{float(effectiveness['volatility_reduction']):.1%}",
            help="Reduction in P&L volatility from hedging"
        )
    
    with col2:
        st.metric(
            "Loss Probability Reduction", 
            f"{float(effectiveness['loss_prob_reduction']):.1%}",
            help="Reduction in probability of loss"
        )
    
    with col3:
        st.metric(
            "Expected P&L Change",
            f"${float(effectiveness['mean_difference']):,.0f}",
            help="Change in expected P&L from hedging"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio Change",
            f"{float(effectiveness['sharpe_improvement']):.3f}",
            delta=f"Hedged: {float(effectiveness['hedged_sharpe']):.3f}",
            help="Improvement in risk-adjusted returns"
        )


def display_risk_metrics(results):
    """Display risk metrics table."""
    
    st.markdown('<h3 class="section-header">Risk Metrics Comparison</h3>', unsafe_allow_html=True)
    
    hedged_risk = results['hedged_risk']
    unhedged_risk = results['unhedged_risk']
    confidence = results['params']['confidence']
    
    # Create comparison table
    risk_comparison = summarize_risk_comparison(hedged_risk, unhedged_risk)
    
    # Format the comparison table
    risk_comparison['Value_Unhedged'] = risk_comparison['Value_Unhedged'].apply(lambda x: f"${x:,.0f}")
    risk_comparison['Value_Hedged'] = risk_comparison['Value_Hedged'].apply(lambda x: f"${x:,.0f}")
    risk_comparison['Difference'] = risk_comparison['Difference'].apply(lambda x: f"${x:,.0f}")
    risk_comparison['Improvement'] = risk_comparison['Improvement'].apply(lambda x: f"{x:.1%}")
    
    # Rename columns
    risk_comparison.columns = ['Risk Metric', 'Unhedged', 'Hedged', 'Difference', 'Improvement']
    
    st.dataframe(risk_comparison, hide_index=True, use_container_width=True)
    
    # Additional metrics
    st.markdown("#### Additional Risk Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_exp = float(results['delta_exposure'])
        st.metric(
            "Delta Exposure",
            f"{delta_exp:,.0f}",
            help="Net delta exposure of the hedged position"
        )
    
    with col2:
        hedge_ratio = float(results['params']['hedge_ratio'])
        st.metric(
            "Hedge Effectiveness",
            f"{hedge_ratio:.1%}",
            help="Percentage of position that is hedged"
        )
    
    with col3:
        # Get Sharpe ratios
        hedged_sharpe_row = hedged_risk[hedged_risk['Metric'] == 'Sharpe Ratio']
        
        if not hedged_sharpe_row.empty:
            hedged_sharpe = float(hedged_sharpe_row['Value'].iloc[0])
            st.metric(
                "Sharpe Ratio (Hedged)",
                f"{hedged_sharpe:.3f}",
                help="Risk-adjusted return measure"
            )
        else:
            st.metric("Sharpe Ratio", "N/A")


def display_stress_testing(results):
    """Display stress testing analysis."""
    
    st.markdown('<h3 class="section-header">Historical Crisis Stress Testing</h3>', unsafe_allow_html=True)
    st.markdown("Test your hedging strategy against major historical market crises.")
    
    # Get current position details
    params = results['params']
    current_price = results['current_price']
    
    # Scenario selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Select Crisis Scenarios to Test")
        selected_scenarios = st.multiselect(
            "Choose scenarios:",
            options=list(STRESS_SCENARIOS.keys()),
            default=list(STRESS_SCENARIOS.keys())[:3],
            help="Select which historical crises to simulate"
        )
    
    with col2:
        st.markdown("#### Quick Analysis")
        if st.button("🚀 Run All Scenarios", type="primary"):
            selected_scenarios = list(STRESS_SCENARIOS.keys())
    
    if selected_scenarios:
        # Run stress tests
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
                # Options provide asymmetric protection
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
        
        # Display results
        st.markdown("#### Stress Test Results")
        results_df = pd.DataFrame(stress_results)
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        
        # Summary
        st.markdown("#### Key Insights")
        unhedged_values = [float(r["Unhedged P&L"].replace("$", "").replace(",", "")) for r in stress_results]
        hedged_values = [float(r["Hedged P&L"].replace("$", "").replace(",", "")) for r in stress_results]
        
        worst_unhedged = min(unhedged_values)
        worst_hedged = min(hedged_values)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Worst Unhedged Loss", f"${worst_unhedged:,.0f}")
        
        with col2:
            st.metric("Worst Hedged Loss", f"${worst_hedged:,.0f}", 
                     delta=f"${worst_hedged - worst_unhedged:,.0f}")
        
        with col3:
            if worst_unhedged < 0:
                risk_reduction = (abs(worst_unhedged) - abs(worst_hedged)) / abs(worst_unhedged)
                st.metric("Risk Reduction", f"{risk_reduction:.1%}")
    
    else:
        st.info("👆 Select at least one crisis scenario to run stress tests.")


# =============================================================================
# PORTFOLIO INTERFACE FUNCTIONS
# =============================================================================

def portfolio_interface():
    """Portfolio management interface."""
    
    st.markdown('<div class="success-banner">✨ <strong>Portfolio Mode Active</strong> - Manage multiple commodity positions with advanced risk analytics</div>', unsafe_allow_html=True)
    
    # Create main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        portfolio_builder_sidebar()
    
    with col2:
        portfolio_dashboard()


def portfolio_builder_sidebar():
    """Portfolio builder sidebar."""
    
    st.markdown("## 🏗️ Portfolio Builder")
    
    # Portfolio summary card
    portfolio = st.session_state.portfolio_manager
    
    if len(portfolio) > 0:
        total_notional = sum(pos.notional_value for pos in portfolio.positions.values())
        st.markdown(f"""
        <div class="portfolio-card">
            <h3>📊 Portfolio Summary</h3>
            <p><strong>Positions:</strong> {len(portfolio)}</p>
            <p><strong>Total Notional:</strong> ${total_notional:,.0f}</p>
            <p><strong>Commodities:</strong> {len(set(pos.commodity for pos in portfolio.positions.values()))}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick portfolio templates
    st.markdown("### 🚀 Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Sample Portfolio", help="Load a sample diversified portfolio"):
            st.session_state.portfolio_manager = create_sample_portfolio()
            st.success("Sample portfolio loaded!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", help="Clear all positions"):
            st.session_state.portfolio_manager.clear()
            st.session_state.simulation_run = False
            st.success("Portfolio cleared!")
            st.rerun()
    
    # Add new position form
    st.markdown("### ➕ Add New Position")
    
    with st.form("add_position_form"):
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
                "Position Size:",
                min_value=-100000.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0,
                help="Positive = Long, Negative = Short"
            )
        
        with col2:
            hedge_ratio = st.slider(
                "Hedge Ratio:",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                format="%.2f",
                help="Percentage of position to hedge"
            )
        
        strategy = st.selectbox(
            "Strategy:",
            options=["Futures", "Options"],
            help="Hedging instrument type"
        )
        
        strike_price = None
        if strategy == "Options":
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
            except:
                strike_price = st.number_input(
                    "Strike Price:",
                    value=75.0,
                    help="Option strike price"
                )
        
        submitted = st.form_submit_button("🔥 Add Position", type="primary")
        
        if submitted and position_name and position_name not in st.session_state.portfolio_manager.positions:
            # Create new position
            new_position = Position(
                commodity=commodity,
                size=position_size,
                hedge_ratio=hedge_ratio,
                strategy=strategy,
                strike_price=strike_price
            )
            
            # Add to portfolio
            st.session_state.portfolio_manager.add_position(position_name, new_position)
            st.session_state.simulation_run = False
            st.success(f"✅ Added {position_name} to portfolio!")
            st.rerun()
        
        elif submitted and position_name in st.session_state.portfolio_manager.positions:
            st.error(f"Position '{position_name}' already exists!")
    
    # Current positions list
    if len(portfolio) > 0:
        st.markdown("### 📋 Current Positions")
        
        for name, position in portfolio.positions.items():
            direction_emoji = "📈" if position.size > 0 else "📉"
            hedge_emoji = "🛡️" if position.is_hedged else "⚠️"
            
            st.markdown(f"""
            <div class="position-card">
                <strong>{direction_emoji} {name}</strong><br>
                <small>{position.commodity} • {abs(position.size):,.0f} • {position.hedge_ratio:.1%} hedged {hedge_emoji}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"❌", key=f"remove_{name}", help=f"Remove {name}"):
                st.session_state.portfolio_manager.remove_position(name)
                st.session_state.simulation_run = False
                st.rerun()
    
    # Analysis controls
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
        "Simulations:",
        options=[1000, 5000, 10000],
        index=1
    )
    
    # Update portfolio config
    st.session_state.portfolio_manager.set_config(
        confidence_level=confidence_level,
        simulation_runs=n_simulations
    )
    
    # Run analysis button
    st.markdown("---")
    if st.button("🚀 Analyze Portfolio", type="primary", disabled=len(portfolio) == 0):
        if len(portfolio) > 0:
            st.session_state.simulation_run = True
            st.rerun()
        else:
            st.warning("Add at least one position to analyze!")


def portfolio_dashboard():
    """Main portfolio dashboard."""
    
    portfolio = st.session_state.portfolio_manager
    
    if len(portfolio) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👈 Start by adding positions</h2>
            <p>Use the portfolio builder on the left to add commodity positions</p>
            <p>Try the "Sample Portfolio" button for a quick start!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Calculate portfolio analytics
    with st.spinner("Calculating portfolio analytics..."):
        try:
            portfolio.calculate_correlations().calculate_portfolio_risk()
            analysis_ready = True
        except Exception as e:
            st.error(f"Error calculating analytics: {e}")
            analysis_ready = False
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Overview", 
        "🔗 Correlations", 
        "⚠️ Risk Analysis", 
        "📈 Performance", 
        "🧪 Stress Testing"
    ])
    
    with tab1:
        portfolio_overview_tab(portfolio, analysis_ready)
    
    with tab2:
        correlations_tab(portfolio, analysis_ready)
    
    with tab3:
        portfolio_risk_analysis_tab(portfolio, analysis_ready)
    
    with tab4:
        portfolio_performance_tab(portfolio, analysis_ready)
    
    with tab5:
        portfolio_stress_testing_tab(portfolio, analysis_ready)


def portfolio_overview_tab(portfolio, analysis_ready):
    """Portfolio overview tab."""
    
    st.markdown("### 📊 Portfolio Composition")
    
    # Portfolio summary table
    summary_df = portfolio.get_portfolio_summary()
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Portfolio visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Position weights pie chart
        weights = portfolio.get_portfolio_weights()
        if weights:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.4,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig_pie.update_layout(
                title="Portfolio Allocation by Position",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Commodity exposure
        exposure_df = portfolio.get_commodity_exposure()
        if not exposure_df.empty:
            st.markdown("**Net Commodity Exposure:**")
            st.dataframe(exposure_df, use_container_width=True, hide_index=True)
        
        # Risk metrics summary
        if analysis_ready:
            risk_summary = portfolio.get_portfolio_risk_summary()
            if risk_summary:
                st.markdown("**Portfolio Risk Metrics:**")
                for metric, value in list(risk_summary.items())[:4]:
                    st.metric(metric, value)


def correlations_tab(portfolio, analysis_ready):
    """Correlations analysis tab."""
    
    st.markdown("### 🔗 Cross-Commodity Correlations")
    
    if not analysis_ready:
        st.warning("Run portfolio analysis to see correlations")
        return
    
    corr_matrix = portfolio.get_correlation_matrix()
    
    if corr_matrix.empty:
        st.info("Need at least 2 different commodities to calculate correlations")
        return
    
    # Correlation heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig_heatmap.update_layout(
        title="Commodity Correlation Matrix",
        height=500,
        xaxis_title="Commodities",
        yaxis_title="Commodities"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Correlation insights
    st.markdown("### 📋 Correlation Insights")
    
    correlations = []
    commodities = corr_matrix.columns.tolist()
    
    for i, commodity1 in enumerate(commodities):
        for j, commodity2 in enumerate(commodities):
            if i < j:
                corr_value = corr_matrix.loc[commodity1, commodity2]
                correlations.append({
                    'Commodity Pair': f"{commodity1} vs {commodity2}",
                    'Correlation': f"{corr_value:.3f}",
                    'Relationship': get_correlation_description(corr_value)
                })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)


def portfolio_risk_analysis_tab(portfolio, analysis_ready):
    """Portfolio risk analysis tab."""
    
    st.markdown("### ⚠️ Portfolio Risk Analysis")
    
    if not analysis_ready:
        st.warning("Run portfolio analysis to see risk metrics")
        return
    
    risk_summary = portfolio.get_portfolio_risk_summary()
    
    if not risk_summary:
        st.error("Unable to calculate risk metrics")
        return
    
    # Risk metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_list = list(risk_summary.items())
    
    with col1:
        st.metric(metrics_list[0][0], metrics_list[0][1])
        if len(metrics_list) > 4:
            st.metric(metrics_list[4][0], metrics_list[4][1])
    
    with col2:
        st.metric(metrics_list[1][0], metrics_list[1][1])
        if len(metrics_list) > 5:
            st.metric(metrics_list[5][0], metrics_list[5][1])
    
    with col3:
        st.metric(metrics_list[2][0], metrics_list[2][1])
        if len(metrics_list) > 6:
            st.metric(metrics_list[6][0], metrics_list[6][1])
    
    with col4:
        st.metric(metrics_list[3][0], metrics_list[3][1])
        if len(metrics_list) > 7:
            st.metric(metrics_list[7][0], metrics_list[7][1])
    
    # Portfolio P&L distribution
    st.markdown("### 📊 Portfolio P&L Distribution")
    
    try:
        portfolio_pnl = portfolio._simulate_portfolio_pnl()
        
        if len(portfolio_pnl) > 0:
            fig_hist = go.Figure(data=[go.Histogram(
                x=portfolio_pnl,
                nbinsx=50,
                name='Portfolio P&L',
                marker_color='skyblue',
                opacity=0.7
            )])
            
            # Add VaR line
            var_95 = np.percentile(portfolio_pnl, 5)
            fig_hist.add_vline(
                x=var_95,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VaR (95%): ${var_95:,.0f}"
            )
            
            fig_hist.update_layout(
                title="Portfolio P&L Distribution (Monte Carlo Simulation)",
                xaxis_title="P&L ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not generate P&L distribution: {e}")


def portfolio_performance_tab(portfolio, analysis_ready):
    """Portfolio performance analysis tab."""
    
    st.markdown("### 📈 Portfolio Performance Analysis")
    
    if not analysis_ready:
        st.warning("Run portfolio analysis to see performance metrics")
        return
    
    st.info("🚧 Advanced performance analytics coming soon!")
    st.markdown("""
    **Planned Features:**
    - Historical performance backtesting
    - Hedge effectiveness tracking
    - Performance attribution by position
    - Benchmark comparisons
    - Rolling risk metrics
    """)


def portfolio_stress_testing_tab(portfolio, analysis_ready):
    """Portfolio stress testing tab."""
    
    st.markdown("### 🧪 Portfolio Stress Testing")
    
    if not analysis_ready:
        st.warning("Run portfolio analysis to see stress testing")
        return
    
    st.info("🚧 Multi-commodity stress testing coming soon!")
    st.markdown("""
    **Planned Features:**
    - Historical crisis scenarios (2008, 2020, etc.)
    - Custom scenario builder
    - Component stress testing
    - Correlation breakdown analysis
    - Recovery scenarios
    """)


def get_correlation_description(correlation):
    """Get human-readable correlation description."""
    if correlation > 0.7:
        return "Strong Positive"
    elif correlation > 0.3:
        return "Moderate Positive"
    elif correlation > -0.3:
        return "Weak/No Correlation"
    elif correlation > -0.7:
        return "Moderate Negative"
    else:
        return "Strong Negative"


if __name__ == "__main__":
    main()
