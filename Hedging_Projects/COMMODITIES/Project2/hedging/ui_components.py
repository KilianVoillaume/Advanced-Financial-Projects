"""
hedging/ui_components.py

Enhanced UI components for multi-commodity support
"""

import streamlit as st
from typing import Dict, List
from .data import (
    get_available_commodities_by_category, 
    get_commodity_specs, 
    get_commodity_category
)


def render_enhanced_commodity_selector(key_suffix: str = "") -> str:
    """Enhanced commodity selector with categories"""
    
    # Get categories
    categories = get_available_commodities_by_category()
    
    # Category selection
    selected_category = st.selectbox(
        "Commodity Category:",
        options=list(categories.keys()),
        index=0,
        key=f"category_{key_suffix}",
        help="Select commodity category"
    )
    
    # Commodity selection within category
    commodities_in_category = categories[selected_category]
    
    selected_commodity = st.selectbox(
        "Specific Commodity:",
        options=commodities_in_category,
        index=0,
        key=f"commodity_{key_suffix}",
        help=f"Select {selected_category.lower()} commodity"
    )
    
    # Display commodity specs in a compact way
    specs = get_commodity_specs(selected_commodity)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"**Unit:** {specs['unit']}")
    with col2:
        st.caption(f"**Volatility:** {specs['typical_volatility']:.1%}")
    with col3:
        st.caption(f"**Seasonality:** {specs['seasonality']}")
    
    return selected_commodity


def render_commodity_info_panel(commodity: str):
    """Render information panel for selected commodity"""
    
    specs = get_commodity_specs(commodity)
    category = get_commodity_category(commodity)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%); 
                padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
        <div style="font-weight: 600; color: #2d3748; margin-bottom: 0.5rem;">
            📊 {commodity} Specifications
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
            <div><strong>Category:</strong> {category}</div>
            <div><strong>Contract Size:</strong> {specs.get('contract_size', 'N/A'):,}</div>
            <div><strong>Typical Vol:</strong> {specs['typical_volatility']:.1%}</div>
            <div><strong>Seasonality:</strong> {specs['seasonality']}</div>
            <div><strong>Storage Cost:</strong> {specs['storage_cost']:.1%}</div>
            <div><strong>Pricing Unit:</strong> {specs['unit']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_simple_commodity_selector(key_suffix: str = "", show_all: bool = False) -> str:
    """Simple flat commodity selector (for backward compatibility)"""
    
    if show_all:
        from .data import get_all_available_commodities
        commodities = get_all_available_commodities()
    else:
        # Default to energy commodities for existing functionality
        commodities = ["WTI Crude Oil", "Brent Crude Oil", "Natural Gas"]
    
    selected_commodity = st.selectbox(
        "Select Commodity:",
        options=commodities,
        index=0,
        key=f"simple_commodity_{key_suffix}",
        help="Select commodity for analysis"
    )
    
    return selected_commodity