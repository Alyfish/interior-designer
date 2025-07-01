"""
Smart Search UI Components - Enhanced visibility for intelligent search features
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SmartSearchUI:
    """UI components for displaying smart search insights and features"""
    
    @staticmethod
    def display_search_insights(search_metadata: Dict[str, Any], container=None):
        """
        Display smart search insights showing how the AI enhanced the search
        
        Args:
            search_metadata: Dictionary containing search enhancement details
            container: Streamlit container to display in
        """
        if container is None:
            container = st
            
        with container.expander("🧠 Smart Search Insights", expanded=True):
            # Search Method Used
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_method = search_metadata.get('search_method', 'unknown')
                method_icon = {
                    'hybrid': '🔄',
                    'visual': '👁️',
                    'text': '📝'
                }.get(search_method, '❓')
                st.metric(
                    "Search Method",
                    f"{method_icon} {search_method.title()}",
                    help="Hybrid combines visual and text search for best results"
                )
            
            with col2:
                st.metric(
                    "Products Found",
                    search_metadata.get('total_products', 0),
                    delta=f"From {search_metadata.get('sources_used', 0)} sources"
                )
            
            with col3:
                st.metric(
                    "Smart Enhancements",
                    search_metadata.get('enhancements_applied', 0),
                    help="Number of AI improvements applied to search"
                )
            
            # Query Transformation
            if 'query_transformation' in search_metadata:
                st.markdown("### 🔄 Query Enhancement")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Original:**")
                    st.code(search_metadata['query_transformation'].get('original', 'N/A'))
                
                with col2:
                    st.markdown("**AI Enhanced:**")
                    enhanced = search_metadata['query_transformation'].get('enhanced', 'N/A')
                    st.info(enhanced)
            
            # Style Matching Details
            if 'style_analysis' in search_metadata:
                st.markdown("### 🎨 Style Intelligence")
                style_data = search_metadata['style_analysis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Detected Room Style:**")
                    st.write(f"• {style_data.get('room_style', 'Not detected').title()}")
                    
                with col2:
                    st.markdown("**Compatible Styles:**")
                    for style in style_data.get('compatible_styles', [])[:3]:
                        st.write(f"• {style}")
            
            # Location Information
            if 'location' in search_metadata:
                st.markdown("### 📍 Location-Based Search")
                loc = search_metadata['location']
                st.write(f"Searching for products available in: **{loc.get('city', 'Unknown')}, {loc.get('country', 'Unknown')}**")
    
    @staticmethod
    def display_product_badges(product: Dict[str, Any]) -> str:
        """
        Generate HTML badges for smart product features
        
        Args:
            product: Product dictionary with smart features
            
        Returns:
            HTML string with badges
        """
        badges = []
        
        # Style Match Badge
        style_score = product.get('style_score', 0)
        if style_score > 0.8:
            badges.append('<span style="background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px;">✨ Best Style Match</span>')
        
        # Color Harmony Badge
        color_score = product.get('color_score', 0)
        if color_score > 0.8:
            badges.append('<span style="background-color: #2196F3; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px;">🎨 Color Harmony</span>')
        
        # Size Appropriate Badge
        size_score = product.get('size_score', 0)
        if size_score > 0.8:
            badges.append('<span style="background-color: #FF9800; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px;">📏 Size Appropriate</span>')
        
        # Local Availability Badge
        if product.get('local_availability'):
            badges.append('<span style="background-color: #9C27B0; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px;">📍 Available Locally</span>')
        
        # Good Deal Badge
        if product.get('price_analysis', {}).get('is_good_deal'):
            badges.append('<span style="background-color: #F44336; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px;">💰 Good Deal</span>')
        
        return ' '.join(badges)
    
    @staticmethod
    def display_search_debug_info(debug_data: Dict[str, Any]):
        """
        Display detailed debug information for developers
        
        Args:
            debug_data: Dictionary containing debug information
        """
        with st.expander("🔧 Debug Information", expanded=False):
            # Query Construction Process
            if 'query_construction' in debug_data:
                st.markdown("### Query Construction")
                st.json(debug_data['query_construction'])
            
            # Score Calculations
            if 'score_calculations' in debug_data:
                st.markdown("### Score Calculations")
                for obj_id, scores in debug_data['score_calculations'].items():
                    st.write(f"**Object {obj_id}:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Style", f"{scores.get('style_score', 0):.2f}")
                    with col2:
                        st.metric("Color", f"{scores.get('color_score', 0):.2f}")
                    with col3:
                        st.metric("Size", f"{scores.get('size_score', 0):.2f}")
                    with col4:
                        st.metric("Combined", f"{scores.get('combined_score', 0):.2f}")
            
            # API Responses
            if 'api_responses' in debug_data:
                st.markdown("### API Response Summary")
                for api, response in debug_data['api_responses'].items():
                    st.write(f"**{api}:** {response.get('status', 'Unknown')} - {response.get('products_found', 0)} products")
            
            # Export Debug Data
            if st.button("📥 Export Debug Data", key="export_debug"):
                debug_json = json.dumps(debug_data, indent=2)
                st.download_button(
                    label="Download debug.json",
                    data=debug_json,
                    file_name=f"search_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    @staticmethod
    def create_location_selector() -> Tuple[str, str]:
        """
        Create location selector UI in sidebar
        
        Returns:
            Tuple of (country_code, city)
        """
        st.sidebar.markdown("### 📍 Shopping Location")
        
        # Country selector with common e-commerce countries
        countries = {
            "United States": "us",
            "United Kingdom": "uk", 
            "Canada": "ca",
            "Australia": "au",
            "Germany": "de",
            "France": "fr",
            "Italy": "it",
            "Spain": "es",
            "Japan": "jp",
            "India": "in",
            "Brazil": "br",
            "Mexico": "mx"
        }
        
        selected_country = st.sidebar.selectbox(
            "Country",
            options=list(countries.keys()),
            index=0,
            help="Products will be filtered by availability in this country"
        )
        
        # City input
        city = st.sidebar.text_input(
            "City (Optional)",
            placeholder="e.g., New York, London",
            help="For more precise local availability"
        )
        
        # Show selected location
        country_code = countries[selected_country]
        if city:
            st.sidebar.info(f"🛍️ Shopping in: {city}, {selected_country}")
        else:
            st.sidebar.info(f"🛍️ Shopping in: {selected_country}")
        
        return country_code, city
    
    @staticmethod
    def display_price_intelligence(product: Dict[str, Any]):
        """
        Display price analysis and intelligence
        
        Args:
            product: Product dictionary with price analysis
        """
        price_data = product.get('price_analysis', {})
        
        if not price_data:
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_price = product.get('price', 'N/A')
            st.metric("Current Price", current_price)
        
        with col2:
            avg_price = price_data.get('market_average', 'N/A')
            if avg_price != 'N/A' and current_price != 'N/A':
                try:
                    current_val = float(current_price.replace('$', '').replace(',', ''))
                    avg_val = float(avg_price.replace('$', '').replace(',', ''))
                    delta = ((current_val - avg_val) / avg_val) * 100
                    st.metric("Market Average", avg_price, delta=f"{delta:+.1f}%")
                except:
                    st.metric("Market Average", avg_price)
            else:
                st.metric("Market Average", avg_price)
        
        with col3:
            price_rating = price_data.get('value_rating', 'N/A')
            st.metric("Value Rating", price_rating)
    
    @staticmethod
    def display_material_matching(product: Dict[str, Any], room_materials: List[str]):
        """
        Display material compatibility information
        
        Args:
            product: Product dictionary
            room_materials: List of materials detected in room
        """
        product_materials = product.get('materials', [])
        
        if not product_materials:
            return
        
        st.markdown("**Material Compatibility:**")
        
        compatible_materials = []
        for mat in product_materials:
            if any(room_mat in mat.lower() for room_mat in room_materials):
                compatible_materials.append(f"✅ {mat}")
            else:
                compatible_materials.append(f"• {mat}")
        
        st.write(" | ".join(compatible_materials))