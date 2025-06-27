"""
Product filtering utilities for the interior designer app.
Provides intelligent filtering and deduplication of product results.
"""

import re
import statistics
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def extract_price_value(price_str: str) -> Optional[float]:
    """
    Extract numeric price value from price string.
    
    Args:
        price_str: Price string like "$299.99", "299.99", "From $199"
    
    Returns:
        Float price value or None if not found
    """
    if not price_str or not price_str.strip() or "not available" in price_str.lower():
        return None
    
    # Remove common prefixes and suffixes
    cleaned = re.sub(r'[^\d.,]', '', price_str)
    
    # Handle different price formats
    if ',' in cleaned:  # Remove thousands separators
        cleaned = cleaned.replace(',', '')
    
    try:
        return float(cleaned)
    except ValueError:
        # Only log warnings for unexpected price formats, not "Price not available"
        if "not available" not in price_str.lower():
            logger.debug(f"Could not parse price format: {price_str}")
        return None

def price_band_filter(items: List[Dict[str, Any]], 
                     user_budget_hint: Optional[float] = None,
                     tolerance_factor: float = 3.0) -> List[Dict[str, Any]]:
    """
    Filter items based on price range to remove outliers.
    
    Args:
        items: List of product items
        user_budget_hint: Optional user budget preference
        tolerance_factor: How much variation to allow (default 3x)
    
    Returns:
        Filtered list of items
    """
    if not items:
        return items
    
    # Extract valid prices
    valid_items = []
    prices = []
    
    for item in items:
        price_val = extract_price_value(item.get('price', ''))
        if price_val and price_val > 0:
            item['price_val'] = price_val
            prices.append(price_val)
            valid_items.append(item)
        else:
            # Keep items without valid prices but mark them
            item['price_val'] = None
            valid_items.append(item)
    
    if len(prices) < 3:
        logger.info("Not enough valid prices for filtering, returning all items")
        return valid_items
    
    # Determine target price
    if user_budget_hint:
        target_price = user_budget_hint
        logger.info(f"Using user budget hint: ${target_price}")
    else:
        target_price = statistics.median(prices)
        logger.info(f"Using median price: ${target_price}")
    
    # Calculate price range
    min_price = target_price / tolerance_factor
    max_price = target_price * tolerance_factor
    
    logger.info(f"Price filter range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Filter items
    filtered_items = []
    for item in valid_items:
        price_val = item.get('price_val')
        if price_val is None:
            # Keep items without prices
            filtered_items.append(item)
        elif min_price <= price_val <= max_price:
            filtered_items.append(item)
        else:
            logger.debug(f"Filtered out item '{item.get('title', 'Unknown')}' with price ${price_val}")
    
    logger.info(f"Price filtering: {len(valid_items)} -> {len(filtered_items)} items")
    return filtered_items

def dedupe_by_content(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate products based on title similarity and store.
    
    Args:
        items: List of product items
    
    Returns:
        Deduplicated list of items
    """
    if not items:
        return items
    
    seen_keys = set()
    kept_items = []
    
    for item in items:
        title = item.get('title', '').lower().strip()
        url = item.get('url', '')
        
        # Extract domain from URL
        try:
            domain = urlparse(url).netloc
        except:
            domain = 'unknown'
        
        # Create a key from title and domain
        # Normalize title by removing common words and punctuation
        normalized_title = re.sub(r'\W+', ' ', title)
        words = [word for word in normalized_title.split() if len(word) > 3]
        
        # Use first 5 significant words + domain as key
        key_parts = sorted(words[:5]) + [domain]
        key = '|'.join(key_parts)
        
        if key not in seen_keys:
            seen_keys.add(key)
            kept_items.append(item)
        else:
            logger.debug(f"Deduplicated item: '{title}' from {domain}")
    
    logger.info(f"Deduplication: {len(items)} -> {len(kept_items)} items")
    return kept_items

def validate_product_result(item: Dict[str, Any]) -> bool:
    """
    Validate that a product result has minimum required fields.
    
    Args:
        item: Product item dictionary
    
    Returns:
        True if item is valid, False otherwise
    """
    required_fields = ['title', 'url']
    
    # Check required fields
    for field in required_fields:
        if not item.get(field):
            return False
    
    # Validate URL format
    url = item.get('url', '')
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Validate title length
    title = item.get('title', '')
    if len(title.strip()) < 3:
        return False
    
    return True

def filter_and_validate_results(items: List[Dict[str, Any]], 
                               user_budget_hint: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Apply all filters and validation to product results.
    
    Args:
        items: Raw product results
        user_budget_hint: Optional user budget preference
    
    Returns:
        Filtered and validated results
    """
    if not items:
        return items
    
    logger.info(f"Starting with {len(items)} raw results")
    
    # Step 1: Validate results
    valid_items = [item for item in items if validate_product_result(item)]
    logger.info(f"Validation: {len(items)} -> {len(valid_items)} items")
    
    # Step 2: Price filtering
    price_filtered = price_band_filter(valid_items, user_budget_hint)
    
    # Step 3: Deduplication
    final_items = dedupe_by_content(price_filtered)
    
    logger.info(f"Final filtered results: {len(final_items)} items")
    return final_items

def enhance_furniture_query(base_query: str, style_info: Dict[str, Any]) -> str:
    """
    Enhance furniture search query with style information.
    
    Args:
        base_query: Base search query
        style_info: Dictionary with style, material, color info
    
    Returns:
        Enhanced search query
    """
    enhancements = []
    
    # Add style information
    if style_info.get('style'):
        enhancements.append(style_info['style'])
    
    # Add material information
    if style_info.get('material'):
        enhancements.append(style_info['material'])
    
    # Add color information
    if style_info.get('colour'):
        enhancements.append(style_info['colour'])
    
    # Add era information
    if style_info.get('era'):
        enhancements.append(style_info['era'])
    
    if enhancements:
        enhanced_query = f"{base_query} {' '.join(enhancements)} furniture"
        logger.info(f"Enhanced query: '{base_query}' -> '{enhanced_query}'")
        return enhanced_query
    
    return base_query 