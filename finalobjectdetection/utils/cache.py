"""
Cache utilities for the interior designer app.
Provides LRU caching for expensive operations like API calls.
"""

import logging
from functools import lru_cache
from typing import Optional, Dict, Any
import hashlib
import json

logger = logging.getLogger(__name__)

# Cache for product searches
@lru_cache(maxsize=32)
def cached_product_search(query: str) -> str:
    """
    Cached wrapper for product search to reduce SerpAPI calls.

    Args:
        query: Search query string
    Returns:
        Product search results as string (formatted list)
    """
    from new_product_matcher import search_products_serpapi_tool
    from config import SERP_API_KEY
    logger.info(f"Cache miss for query: '{query}'")
    return search_products_serpapi_tool(query, SERP_API_KEY)

def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a consistent cache key from arguments.
    Useful for complex objects that can't be directly cached.
    """
    # Convert to JSON string and hash it
    key_data = json.dumps((args, sorted(kwargs.items())), sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()

# Cache for image embeddings
@lru_cache(maxsize=16)
def cached_image_embedding(image_path: str) -> Optional[Any]:
    """
    Cached wrapper for image embedding extraction.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Cached embedding or None if extraction fails
    """
    try:
        from vision_features import extract_clip_embedding
        logger.info(f"Cache miss for image embedding: {image_path}")
        return extract_clip_embedding(image_path)
    except Exception as e:
        logger.warning(f"Failed to extract embedding for {image_path}: {e}")
        return None

@lru_cache(maxsize=32)
def cached_reverse_image_search(image_path: str, query_text: str = "") -> list:
    """
    Cached wrapper for the reverse image search to reduce API calls and image uploads.
    The cache key is based on the image path and the query text.

    Args:
        image_path: Path to the local image file.
        query_text: Optional text to refine the visual search.

    Returns:
        A list of product dictionaries from the search results, or an empty list.
    """
    from new_product_matcher import search_products_reverse_image_serpapi, parse_agent_response_to_products
    from config import SERP_API_KEY
    
    logger.info(f"Cache miss for reverse image search on: '{image_path}' with query '{query_text}'")
    
    try:
        # This function performs the actual search, including uploading the image
        api_response_str = search_products_reverse_image_serpapi(
            image_path=image_path,
            serp_api_key=SERP_API_KEY,
            query_text=query_text
        )
        
        if not api_response_str:
            return []
            
        # The API returns a string, so we need to parse it into a list of products
        products = parse_agent_response_to_products(api_response_str)
        return products
        
    except Exception as e:
        logger.error(f"Error during cached reverse image search for {image_path}: {e}", exc_info=True)
        return []

def clear_all_caches():
    """Clear all LRU caches. Useful for testing or memory management."""
    cached_product_search.cache_clear()
    cached_image_embedding.cache_clear()
    cached_reverse_image_search.cache_clear()
    logger.info("All caches cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about cache usage."""
    reverse_search_info = cached_reverse_image_search.cache_info()
    return {
        "product_search": {
            "hits": cached_product_search.cache_info().hits,
            "misses": cached_product_search.cache_info().misses,
            "maxsize": cached_product_search.cache_info().maxsize,
            "currsize": cached_product_search.cache_info().currsize
        },
        "image_embedding": {
            "hits": cached_image_embedding.cache_info().hits,
            "misses": cached_image_embedding.cache_info().misses,
            "maxsize": cached_image_embedding.cache_info().maxsize,
            "currsize": cached_image_embedding.cache_info().currsize
        },
        "reverse_image_search": {
            "hits": reverse_search_info.hits,
            "misses": reverse_search_info.misses,
            "maxsize": reverse_search_info.maxsize,
            "currsize": reverse_search_info.currsize
        }
    } 