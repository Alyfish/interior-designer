"""
Session Tracker - Tracks user behavior within current session only
No database or persistent storage required
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class SessionTracker:
    """Tracks user interactions within current Streamlit session"""
    
    @staticmethod
    def init_session_state():
        """Initialize session tracking in Streamlit session state"""
        if 'interaction_history' not in st.session_state:
            st.session_state.interaction_history = []
        
        if 'style_preferences' not in st.session_state:
            st.session_state.style_preferences = {}
        
        if 'viewed_products' not in st.session_state:
            st.session_state.viewed_products = []
        
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
        
        if 'room_context_history' not in st.session_state:
            st.session_state.room_context_history = []
    
    @staticmethod
    def track_room_upload(room_context: Dict):
        """Track when user uploads a room"""
        try:
            event = {
                'type': 'room_upload',
                'timestamp': datetime.now().isoformat(),
                'room_type': room_context.get('room_type'),
                'brightness': room_context.get('room_brightness'),
                'object_count': room_context.get('object_density', 0)
            }
            st.session_state.interaction_history.append(event)
            
            # Store room context for later reference
            st.session_state.room_context_history.append(room_context)
            
            # Update style preferences
            room_type = room_context.get('room_type')
            if room_type:
                current_count = st.session_state.style_preferences.get(room_type, 0)
                st.session_state.style_preferences[room_type] = current_count + 1
                
        except Exception as e:
            logger.warning(f"Failed to track room upload: {e}")
    
    @staticmethod
    def track_object_selection(object_data: Dict):
        """Track when user selects an object"""
        try:
            event = {
                'type': 'object_selection',
                'timestamp': datetime.now().isoformat(),
                'object_class': object_data.get('class'),
                'object_style': object_data.get('caption_data', {}).get('style', 'unknown')
            }
            st.session_state.interaction_history.append(event)
        except Exception as e:
            logger.warning(f"Failed to track object selection: {e}")
    
    @staticmethod
    def track_product_view(product: Dict, object_class: str):
        """Track when user views a product"""
        try:
            view_data = {
                'timestamp': datetime.now().isoformat(),
                'product_title': product.get('title'),
                'product_price': product.get('price'),
                'object_class': object_class,
                'style_score': product.get('style_score', 0),
                'context_score': product.get('context_score', 0)
            }
            st.session_state.viewed_products.append(view_data)
            
            # Keep only last 50 viewed products
            if len(st.session_state.viewed_products) > 50:
                st.session_state.viewed_products = st.session_state.viewed_products[-50:]
                
        except Exception as e:
            logger.warning(f"Failed to track product view: {e}")
    
    @staticmethod
    def get_session_insights() -> Dict:
        """Get insights from current session"""
        try:
            insights = {
                'session_duration': (datetime.now() - st.session_state.session_start).seconds,
                'interactions_count': len(st.session_state.interaction_history),
                'products_viewed': len(st.session_state.viewed_products),
                'preferred_room_types': st.session_state.style_preferences,
                'avg_style_score': 0,
                'avg_context_score': 0,
                'rooms_analyzed': len(st.session_state.room_context_history)
            }
            
            # Calculate average scores
            if st.session_state.viewed_products:
                style_scores = [p.get('style_score', 0) for p in st.session_state.viewed_products if p.get('style_score') is not None]
                context_scores = [p.get('context_score', 0) for p in st.session_state.viewed_products if p.get('context_score') is not None]
                
                if style_scores:
                    insights['avg_style_score'] = sum(style_scores) / len(style_scores)
                if context_scores:
                    insights['avg_context_score'] = sum(context_scores) / len(context_scores)
            
            return insights
            
        except Exception as e:
            logger.warning(f"Failed to get session insights: {e}")
            return {
                'session_duration': 0,
                'interactions_count': 0,
                'products_viewed': 0,
                'preferred_room_types': {},
                'avg_style_score': 0,
                'avg_context_score': 0,
                'rooms_analyzed': 0
            }
    
    @staticmethod
    def boost_similar_products(products: List[Dict]) -> List[Dict]:
        """Boost products similar to previously viewed ones"""
        if not products or not st.session_state.viewed_products:
            return products
        
        try:
            # Extract viewed product characteristics
            viewed_styles = []
            viewed_classes = []
            viewed_colors = []
            
            for viewed in st.session_state.viewed_products[-10:]:  # Last 10 views
                title = viewed.get('product_title', '').lower()
                
                # Extract style keywords
                for style in ['modern', 'traditional', 'industrial', 'scandinavian', 'bohemian', 'minimalist', 'contemporary']:
                    if style in title:
                        viewed_styles.append(style)
                
                # Extract color keywords
                for color in ['black', 'white', 'gray', 'brown', 'beige', 'blue', 'green', 'red']:
                    if color in title:
                        viewed_colors.append(color)
                        
                viewed_classes.append(viewed.get('object_class'))
            
            # Boost products with similar characteristics
            boosted_products = []
            for product in products:
                boosted_product = product.copy()
                boost = 0.0
                
                product_title = product.get('title', '').lower()
                
                # Boost if style matches viewed styles
                for style in viewed_styles:
                    if style in product_title:
                        boost += 0.1
                
                # Boost if color matches viewed colors
                for color in viewed_colors:
                    if color in product_title:
                        boost += 0.05
                
                # Apply boost to score
                if 'combined_score' in boosted_product:
                    boosted_product['combined_score'] *= (1 + boost)
                    boosted_product['session_boost'] = boost
                elif 'context_score' in boosted_product:
                    boosted_product['context_score'] *= (1 + boost)
                    boosted_product['session_boost'] = boost
                
                boosted_products.append(boosted_product)
            
            # Re-sort by boosted scores
            boosted_products.sort(key=lambda x: x.get('combined_score', x.get('context_score', 0)), reverse=True)
            
            return boosted_products
            
        except Exception as e:
            logger.warning(f"Failed to boost products: {e}")
            return products