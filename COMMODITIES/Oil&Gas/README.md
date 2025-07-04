# 🛢️ Oil & Gas Hedging Simulator

A Streamlit web application for simulating and analyzing hedging strategies on oil & gas commodities. Compare hedged vs unhedged positions with interactive charts and risk metrics.

## What it does

This app helps analyze commodity hedging decisions by:

- **Simulating P&L scenarios** for hedged vs unhedged positions
- **Calculating risk metrics** like VaR, CVaR, and Expected P&L  
- **Showing payoff diagrams** at different price levels
- **Comparing hedging effectiveness** through Monte Carlo simulation

## Features

- **Commodities**: WTI Oil, Brent Oil, Natural Gas
- **Strategies**: Futures hedging and Options hedging
- **Live data**: Real-time prices via Yahoo Finance
- **Interactive charts**: Price history, payoff diagrams, P&L distributions
- **Risk analysis**: Professional risk metrics and comparisons

## Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the app**
```bash
streamlit run app.py
```

3. **Use the app**
- Select commodity and position size
- Choose hedging strategy (Futures or Options)
- Set hedge ratio and risk parameters
- Click "Run Simulation" to see results

## Project Structure

```
Oil-Gas/
├── app.py                 # Main Streamlit app
├── hedging/
│   ├── data.py           # Price data fetching
│   ├── strategies.py     # Hedge calculations
│   ├── simulation.py     # Monte Carlo simulation
│   └── risk.py           # Risk metrics
└── requirements.txt      # Dependencies
```

## How to Use

1. **Select Parameters**: Choose commodity, position size, and hedging strategy
2. **Configure Options**: Set strike price and expiration (for options strategy)
3. **Set Risk Settings**: Choose confidence level and number of simulations
4. **View Results**: Analyze charts, payoff diagrams, and risk metrics

## Example

For a long 1,000 barrel WTI position at $75:
- **Unhedged**: Full exposure to price movements
- **80% Futures Hedge**: Reduces volatility but locks in price
- **80% Put Options**: Downside protection with upside participation

## Deployment

**Streamlit Cloud** (recommended):
1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

**Local deployment**:
```bash
streamlit run app.py
```
