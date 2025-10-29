"""
hedging/stress_testing.py

Historical stress testing scenarios for oil & gas hedging strategies.
Models major market crises and their impact on hedged vs unhedged positions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


STRESS_SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Oil crashed from $147 to $34 in 6 months",
        "oil_peak_to_trough": -0.77,  # -77% peak to trough
        "volatility_multiplier": 2.5,
        "duration_days": 180,
        "gas_correlation": 0.85,  # Natural gas followed oil down
        "timeline": "Jul 2008 - Dec 2008"
    },
    
    "2020 COVID Pandemic": {
        "description": "Oil briefly went negative, unprecedented demand destruction",
        "oil_peak_to_trough": -0.80,  # -80% drop, WTI went negative
        "volatility_multiplier": 3.0,
        "duration_days": 90,
        "gas_correlation": 0.60, 
        "timeline": "Feb 2020 - May 2020"
    },
    
    "2022 Ukraine War": {
        "description": "Supply shock drove oil from $75 to $130",
        "oil_peak_to_trough": 0.73,   # +73% spike
        "volatility_multiplier": 2.0,
        "duration_days": 120,
        "gas_correlation": 1.20, 
        "timeline": "Feb 2022 - Jun 2022"
    },
    
    "2014-2016 Oil Crash": {
        "description": "Shale oil boom caused sustained bear market",
        "oil_peak_to_trough": -0.76,  # -76% over 2 years
        "volatility_multiplier": 1.8,
        "duration_days": 600,  
        "gas_correlation": 0.90,
        "timeline": "Jun 2014 - Feb 2016"
    },
    
    "1990 Gulf War": {
        "description": "Iraq invasion of Kuwait caused oil spike",
        "oil_peak_to_trough": 1.20,   # +120% spike
        "volatility_multiplier": 2.2,
        "duration_days": 60,  
        "gas_correlation": 0.70,
        "timeline": "Aug 1990 - Oct 1990"
    }
}


def apply_stress_scenario(base_prices: pd.Series, scenario_name: str, commodity_type: str = "oil") -> pd.Series:

    if scenario_name not in STRESS_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    scenario = STRESS_SCENARIOS[scenario_name]
    
    price_shock = scenario["oil_peak_to_trough"]
    vol_multiplier = scenario["volatility_multiplier"]
    duration = scenario["duration_days"]
    
    # Adjust for natural gas correlation
    if commodity_type == "gas":
        price_shock *= scenario["gas_correlation"]
    
    stressed_prices = base_prices.copy()
    shock_periods = min(duration, len(base_prices))
    
    for i in range(shock_periods):
        # Gradual application of shock (starts slow, accelerates, then slows)
        progress = i / shock_periods
        shock_factor = price_shock * np.sin(progress * np.pi)  # Sine wave for realistic progression
        
        base_vol = base_prices.pct_change().std()
        extra_vol = np.random.normal(0, base_vol * (vol_multiplier - 1))
        
        # Apply shock and volatility
        if i < len(stressed_prices):
            stressed_prices.iloc[i] = base_prices.iloc[i] * (1 + shock_factor + extra_vol)
    
    # Ensure no negative prices (except for COVID oil scenario)
    if scenario_name != "2020 COVID Pandemic":
        stressed_prices = stressed_prices.clip(lower=0.01)
    
    return stressed_prices


def run_stress_test_portfolio(positions: Dict[str, Dict], scenarios: List[str] = None) -> pd.DataFrame:

    if scenarios is None:
        scenarios = list(STRESS_SCENARIOS.keys())
    
    results = []
    
    for scenario_name in scenarios:
        scenario_result = {
            "Scenario": scenario_name,
            "Description": STRESS_SCENARIOS[scenario_name]["description"],
            "Timeline": STRESS_SCENARIOS[scenario_name]["timeline"]
        }
        
        total_unhedged_pnl = 0
        total_hedged_pnl = 0
        
        for position_name, position in positions.items():
            unhedged_pnl, hedged_pnl = simulate_position_stress(position, scenario_name)
            
            total_unhedged_pnl += unhedged_pnl
            total_hedged_pnl += hedged_pnl
            
            scenario_result[f"{position_name}_Unhedged"] = unhedged_pnl
            scenario_result[f"{position_name}_Hedged"] = hedged_pnl
        
        scenario_result["Total_Unhedged_PnL"] = total_unhedged_pnl
        scenario_result["Total_Hedged_PnL"] = total_hedged_pnl
        scenario_result["Hedge_Benefit"] = total_hedged_pnl - total_unhedged_pnl
        scenario_result["Hedge_Effectiveness"] = (
            abs(scenario_result["Hedge_Benefit"]) / abs(total_unhedged_pnl) 
            if total_unhedged_pnl != 0 else 0
        )
        
        results.append(scenario_result)
    
    return pd.DataFrame(results)


def simulate_position_stress(position: Dict, scenario_name: str) -> Tuple[float, float]:

    scenario = STRESS_SCENARIOS[scenario_name]
    
    commodity = position.get("commodity", "oil")
    commodity_type = "gas" if "gas" in commodity.lower() else "oil"
    
    price_shock = scenario["oil_peak_to_trough"]
    if commodity_type == "gas":
        price_shock *= scenario["gas_correlation"]
    
    position_size = position.get("position_size", 1000)
    current_price = position.get("current_price", 75.0)
    
    price_change = current_price * price_shock
    unhedged_pnl = price_change * position_size
    
    hedge_ratio = position.get("hedge_ratio", 0.0)
    strategy = position.get("strategy", "Futures")
    
    if strategy == "Futures":
        # Futures hedge provides linear protection
        hedge_pnl = -price_change * position_size * hedge_ratio
    elif strategy == "Options":
        # Options provide asymmetric protection
        if price_change < 0:  # Puts protect against downside
            hedge_pnl = -price_change * position_size * hedge_ratio * 0.8  # 80% effectiveness
        else:  # Limited upside protection
            hedge_pnl = -price_change * position_size * hedge_ratio * 0.2
    else:
        hedge_pnl = 0
    
    hedged_pnl = unhedged_pnl + hedge_pnl
    
    return unhedged_pnl, hedged_pnl


def create_stress_test_summary(results_df: pd.DataFrame) -> Dict[str, any]:
    
    summary = {
        "worst_unhedged_scenario": results_df.loc[results_df["Total_Unhedged_PnL"].idxmin(), "Scenario"],
        "worst_unhedged_loss": results_df["Total_Unhedged_PnL"].min(),
        "worst_hedged_scenario": results_df.loc[results_df["Total_Hedged_PnL"].idxmin(), "Scenario"],
        "worst_hedged_loss": results_df["Total_Hedged_PnL"].min(),
        "average_hedge_benefit": results_df["Hedge_Benefit"].mean(),
        "max_hedge_benefit": results_df["Hedge_Benefit"].max(),
        "average_hedge_effectiveness": results_df["Hedge_Effectiveness"].mean(),
        "scenarios_with_positive_hedge": len(results_df[results_df["Hedge_Benefit"] > 0]),
        "total_scenarios_tested": len(results_df)
    }
    
    summary["risk_reduction"] = (
        (abs(summary["worst_unhedged_loss"]) - abs(summary["worst_hedged_loss"])) / 
        abs(summary["worst_unhedged_loss"])
        if summary["worst_unhedged_loss"] != 0 else 0
    )
    
    return summary


def get_scenario_details(scenario_name: str) -> Dict[str, str]:
    
    if scenario_name not in STRESS_SCENARIOS:
        return {}
    
    scenario = STRESS_SCENARIOS[scenario_name].copy()
    
    scenario["price_change_pct"] = f"{scenario['oil_peak_to_trough']:.1%}"
    scenario["volatility_increase"] = f"{(scenario['volatility_multiplier'] - 1) * 100:.0f}%"
    scenario["duration_months"] = f"{scenario['duration_days'] / 30:.1f}"
    
    return scenario


if __name__ == "__main__":
    print("Testing stress scenarios...")
    
    test_position = {
        "commodity": "WTI Cushing",
        "position_size": 10000,  # 10,000 barrels
        "current_price": 75.0,
        "hedge_ratio": 0.75,
        "strategy": "Futures"
    }
    
    print("\nTesting single position stress scenarios:")
    for scenario_name in STRESS_SCENARIOS.keys():
        unhedged, hedged = simulate_position_stress(test_position, scenario_name)
        print(f"{scenario_name}: Unhedged: ${unhedged:,.0f}, Hedged: ${hedged:,.0f}, Benefit: ${hedged-unhedged:,.0f}")
    
    test_portfolio = {
        "WTI_Position": {
            "commodity": "WTI Cushing",
            "position_size": 10000,
            "current_price": 75.0,
            "hedge_ratio": 0.8,
            "strategy": "Futures"
        },
        "Gas_Position": {
            "commodity": "Natural Gas",
            "position_size": 50000,  # MMBtu
            "current_price": 3.5,
            "hedge_ratio": 0.6,
            "strategy": "Options"
        }
    }
    
    print("\nTesting portfolio stress scenarios:")
    portfolio_results = run_stress_test_portfolio(test_portfolio)
    print(portfolio_results[["Scenario", "Total_Unhedged_PnL", "Total_Hedged_PnL", "Hedge_Benefit"]])
    
    summary = create_stress_test_summary(portfolio_results)
    print(f"\nWorst case unhedged: ${summary['worst_unhedged_loss']:,.0f} in {summary['worst_unhedged_scenario']}")
    print(f"Worst case hedged: ${summary['worst_hedged_loss']:,.0f} in {summary['worst_hedged_scenario']}")
    print(f"Average hedge benefit: ${summary['average_hedge_benefit']:,.0f}")
    print(f"Risk reduction: {summary['risk_reduction']:.1%}")
