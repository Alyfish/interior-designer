"""
Enhanced Display Integration - Adds tracking and improvements
This is ADDITIVE - wraps existing display functions without modifying them
"""

import streamlit as st
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def display_all_objects_with_products_enhanced(
    objects: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    container=None
):
    """
    Enhanced version of display_all_objects_with_products that adds tracking.
    Falls back to original if tracking not available.
    
    Args:
        objects: List of processed objects
        search_results: List of search results
        container: Streamlit container
    """
    # Import original function
    from utils.object_product_integration import display_all_objects_with_products
    
    # Try to add tracking
    try:
        from session_tracker import SessionTracker
        
        # Track product views when displaying
        for idx, result in enumerate(search_results):
            if 'products' in result and result['products']:
                object_class = result.get('object_class', 'unknown')
                
                # Track views for top products
                for product in result['products'][:3]:  # Top 3 products
                    SessionTracker.track_product_view(product, object_class)
                    
    except Exception as e:
        logger.debug(f"Product tracking not available: {e}")
    
    # Call original display function
    display_all_objects_with_products(objects, search_results, container)
    
    # Add enhanced features if available
    try:
        # Show style score legend if products have scores
        has_scores = any(
            any(p.get('style_score') is not None for p in result.get('products', []))
            for result in search_results
        )
        
        if has_scores and container:
            with container.expander("📊 Score Legend", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Style Score**: How well the product matches your room's style")
                with col2:
                    st.markdown("**Context Score**: Overall compatibility with your space")
                with col3:
                    st.markdown("**Session Boost**: Based on your viewing patterns")
                
    except Exception as e:
        logger.debug(f"Enhanced display features not available: {e}")