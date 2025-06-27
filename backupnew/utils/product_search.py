"""
Product search utilities for the interior designer app.
"""

import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

def search_furniture_products(query: str, image_path: str = None, 
                            search_method: str = "hybrid") -> List[Dict[str, Any]]:
    """
    Search for furniture products using various methods.
    
    Args:
        query: Text description of the furniture item
        image_path: Optional path to image for visual search
        search_method: "text", "visual", or "hybrid"
    
    Returns:
        List of product results
    """
    try:
        from new_product_matcher import search_products_enhanced
        
        # Use the enhanced product search
        results = search_products_enhanced(
            query=query, 
            image_path=image_path,
            search_method=search_method
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in search_furniture_products: {e}")
        return []

def enhance_search_query(base_query: str, object_info: Dict[str, Any]) -> str:
    """
    Enhance a basic search query with additional object information.
    
    Args:
        base_query: Basic search query
        object_info: Information about the detected object
    
    Returns:
        Enhanced search query
    """
    enhanced_parts = [base_query]
    
    # Add color information if available
    if 'color' in object_info:
        enhanced_parts.append(object_info['color'])
    
    # Add style information if available
    if 'style' in object_info:
        enhanced_parts.append(object_info['style'])
    
    # Add material information if available
    if 'material' in object_info:
        enhanced_parts.append(object_info['material'])
    
    return ' '.join(enhanced_parts)

def filter_product_results(products: List[Dict[str, Any]], 
                         filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Filter product results based on various criteria.
    
    Args:
        products: List of product results
        filters: Dictionary of filter criteria
    
    Returns:
        Filtered list of products
    """
    if not filters:
        return products
    
    filtered_products = []
    
    for product in products:
        include_product = True
        
        # Price range filter
        if 'price_range' in filters:
            min_price, max_price = filters['price_range']
            product_price = product.get('price', 0)
            if isinstance(product_price, str):
                # Try to extract numeric price
                import re
                price_match = re.search(r'[\d.]+', product_price)
                if price_match:
                    product_price = float(price_match.group())
                else:
                    product_price = 0
            
            if not (min_price <= product_price <= max_price):
                include_product = False
        
        # Store filter
        if 'stores' in filters:
            product_store = product.get('store', '').lower()
            if not any(store.lower() in product_store for store in filters['stores']):
                include_product = False
        
        # Rating filter
        if 'min_rating' in filters:
            product_rating = product.get('rating', 0)
            if product_rating < filters['min_rating']:
                include_product = False
        
        if include_product:
            filtered_products.append(product)
    
    return filtered_products

def format_product_for_display(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a product result for display in the UI.
    
    Args:
        product: Raw product data
    
    Returns:
        Formatted product data
    """
    formatted = {
        'title': product.get('title', 'Unknown Product'),
        'price': product.get('price', 'Price not available'),
        'url': product.get('url', '#'),
        'image_url': product.get('image_url', ''),
        'store': product.get('store', 'Unknown Store'),
        'rating': product.get('rating', 0),
        'description': product.get('description', ''),
        'match_score': product.get('match_score', 0.5)
    }
    
    # Clean up price formatting
    if isinstance(formatted['price'], str) and formatted['price'] != 'Price not available':
        # Ensure price starts with $ if it's a number
        import re
        if re.match(r'^\d', formatted['price']):
            formatted['price'] = f"${formatted['price']}"
    
    return formatted

def get_product_recommendations(detected_objects: List[Dict[str, Any]], 
                              user_preferences: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Get product recommendations based on detected objects and user preferences.
    
    Args:
        detected_objects: List of detected furniture objects
        user_preferences: User preferences for filtering/ranking
    
    Returns:
        List of recommended products
    """
    all_recommendations = []
    
    for obj in detected_objects:
        # Create search query for this object
        query = obj.get('class_name', 'furniture')
        
        # Enhance query with object details
        if user_preferences:
            enhanced_query = enhance_search_query(query, user_preferences)
        else:
            enhanced_query = query
        
        # Search for products
        products = search_furniture_products(enhanced_query)
        
        # Add object context to each product
        for product in products:
            product['source_object'] = obj
            product['search_query'] = enhanced_query
        
        all_recommendations.extend(products)
    
    # Remove duplicates based on URL
    unique_products = {}
    for product in all_recommendations:
        url = product.get('url', '')
        if url and url not in unique_products:
            unique_products[url] = product
    
    return list(unique_products.values()) 