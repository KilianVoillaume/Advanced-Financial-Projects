"""
app.py

Main Streamlit application for the Oil & Gas Hedging Simulator.
Simulates and analyzes hedging strategies on oil & gas commodities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our custom modules
from hedging.data import get_prices, get_current_price, get_available_commodities, validate_commodity
from hedging.strategies import compute_payoff_diagram, get_hedge_summary
from hedging.simulation import simulate_hedged_vs_unhedged, compare_hedging_effectiveness
from hedging.risk import calculate_risk_metrics, calculate_delta_exposure, summarize_risk_comparison


# Page configuration
st.set_page_config(
    page_title="Oil & Gas Hedging Simulator",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # App header
    st.markdown('<h1 class="main-header">🛢️ Oil & Gas Hedging Simulator</h1>', unsafe_allow_html=True)
    st.markdown("**Simulate and analyze hedging strategies on oil & gas commodities**")
    
    # Initialize session state
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("## 📊 Simulation Parameters")
        
        # Commodity selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🏭 Commodity Selection")
        commodity = st.selectbox(
            "Select Commodity:",
            options=get_available_commodities(),
            index=0,
            help="Choose the commodity for hedging analysis"
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
        
        # Options-specific parameters (enhanced UI from update)
        strike_price = None
        option_expiry = None
        
        if strategy == "Options":
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Options Parameters")
            
            try:
                current_price = get_current_price(commodity)
                
                # Strike price selection
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
                
                # Option expiration
                option_expiry = st.selectbox(
                    "Option Expiration:",
                    options=[1, 3, 6, 12],
                    index=1,
                    format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
                    help="Time until option expiration"
                )
                
            except Exception as e:
                st.error(f"Error getting current price: {e}")
                strike_price = 75.0  # Default fallback
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
                st.session_state.prices = prices
                st.session_state.current_price = current_price
                st.session_state.sim_results = sim_results
                st.session_state.payoff_data = payoff_data
                st.session_state.hedged_risk = hedged_risk
                st.session_state.unhedged_risk = unhedged_risk
                st.session_state.delta_exposure = delta_exposure
                st.session_state.params = {
                    'commodity': commodity,
                    'position': position,
                    'strategy': strategy,
                    'hedge_ratio': hedge_ratio,
                    'strike_price': strike_price,
                    'confidence': confidence
                }
                
            except Exception as e:
                st.error(f"Error running simulation: {e}")
                st.session_state.simulation_run = False
    
    # Display results if simulation has been run
    if st.session_state.simulation_run and hasattr(st.session_state, 'prices'):
        display_results()


def display_results():
    """Display simulation results and charts."""
    
    params = st.session_state.params
    
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
            f"${st.session_state.current_price:.2f}",
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
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Payoff Diagram", "🎯 P&L Distribution", "📋 Risk Metrics"])
    
    with tab1:
        display_price_chart()
    
    with tab2:
        display_payoff_diagram()
    
    with tab3:
        display_pnl_distribution()
    
    with tab4:
        display_risk_metrics()


def display_price_chart():
    """Display historical price chart."""
    
    st.markdown('<h3 class="section-header">Historical Price Chart</h3>', unsafe_allow_html=True)
    
    prices = st.session_state.prices
    commodity = st.session_state.params['commodity']
    
    # Create interactive price chart with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        mode='lines',
        name=f'{commodity} Price',
        line=dict(color='#1f4e79', width=2)
    ))
    
    # Add current price line
    current_price_val = float(st.session_state.current_price)
    fig.add_hline(
        y=current_price_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: ${current_price_val:.2f}"
    )
    
    # Add strike price line for options
    if st.session_state.params['strategy'] == 'Options' and st.session_state.params['strike_price']:
        strike_price_val = float(st.session_state.params['strike_price'])
        fig.add_hline(
            y=strike_price_val,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Strike: ${strike_price_val:.2f}"
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


def display_payoff_diagram():
    """Display payoff diagram (enhanced feature from update)."""
    
    st.markdown('<h3 class="section-header">Payoff Diagram at Expiry</h3>', unsafe_allow_html=True)
    st.markdown("This diagram shows profit/loss for different price scenarios at expiration.")
    
    payoff_data = st.session_state.payoff_data
    current_price = st.session_state.current_price
    
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
    
    # Add net P&L (most important)
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
    current_price_val = float(current_price)
    fig.add_vline(
        x=current_price_val,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Current Price: ${current_price_val:.2f}"
    )
    
    # Add breakeven points
    for i, breakeven in enumerate(payoff_data['breakeven_prices']):
        breakeven_val = float(breakeven)
        fig.add_vline(
            x=breakeven_val,
            line_dash="dot",
            line_color="purple",
            annotation_text=f"Breakeven: ${breakeven_val:.2f}"
        )
    
    fig.update_layout(
        title="Payoff Diagram: Hedged vs Unhedged Position",
        xaxis_title="Spot Price at Expiry ($)",
        yaxis_title="Profit & Loss ($)",
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
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


def display_pnl_distribution():
    """Display P&L distribution histogram."""
    
    st.markdown('<h3 class="section-header">P&L Distribution Analysis</h3>', unsafe_allow_html=True)
    st.markdown("Monte Carlo simulation results showing probability distribution of outcomes.")
    
    sim_results = st.session_state.sim_results
    
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
    col1, col2, col3 = st.columns(3)
    
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


def display_risk_metrics():
    """Display risk metrics table."""
    
    st.markdown('<h3 class="section-header">Risk Metrics Comparison</h3>', unsafe_allow_html=True)
    
    hedged_risk = st.session_state.hedged_risk
    unhedged_risk = st.session_state.unhedged_risk
    confidence = st.session_state.params['confidence']
    
    # Create comparison table
    risk_comparison = summarize_risk_comparison(hedged_risk, unhedged_risk)
    
    # Format the comparison table for better display
    risk_comparison['Value_Unhedged'] = risk_comparison['Value_Unhedged'].apply(lambda x: f"${x:,.0f}")
    risk_comparison['Value_Hedged'] = risk_comparison['Value_Hedged'].apply(lambda x: f"${x:,.0f}")
    risk_comparison['Difference'] = risk_comparison['Difference'].apply(lambda x: f"${x:,.0f}")
    risk_comparison['Improvement'] = risk_comparison['Improvement'].apply(lambda x: f"{x:.1%}")
    
    # Rename columns for display
    risk_comparison.columns = ['Risk Metric', 'Unhedged', 'Hedged', 'Difference', 'Improvement']
    
    st.dataframe(
        risk_comparison,
        hide_index=True,
        use_container_width=True
    )
    
    # Additional metrics
    st.markdown("#### Additional Risk Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delta_exp_val = float(st.session_state.delta_exposure)
        st.metric(
            "Delta Exposure",
            f"{delta_exp_val:,.0f}",
            help="Net delta exposure of the hedged position"
        )
    
    with col2:
        hedge_ratio_val = float(st.session_state.params['hedge_ratio'])
        st.metric(
            "Hedge Effectiveness",
            f"{hedge_ratio_val:.1%}",
            help="Percentage of position that is hedged"
        )
    
    # Risk interpretation
    with st.expander("📖 Risk Metrics Explanation"):
        st.markdown(f"""
        **Expected P&L**: Average profit/loss from {len(st.session_state.sim_results['hedged_pnl']):,} simulations
        
        **VaR ({confidence:.0%})**: Maximum expected loss at {confidence:.0%} confidence level
        
        **CVaR ({confidence:.0%})**: Average loss in worst {100-confidence*100:.0f}% of scenarios (tail risk)
        
        **Volatility**: Standard deviation of P&L outcomes
        
        **Delta Exposure**: Sensitivity to price changes after hedging
        
        **Improvement**: Positive values indicate hedging reduces risk
        """)


if __name__ == "__main__":
    main()
