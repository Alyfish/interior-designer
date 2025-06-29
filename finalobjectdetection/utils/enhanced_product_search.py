"""
Enhanced Product Search Module

This module provides improved product search functionality with better query construction,
visual search integration, and result ranking - all while preserving existing features.
"""

import logging
import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logger = logging.getLogger(__name__)


class EnhancedProductSearcher:
    """Enhanced product search with improved query construction and ranking."""
    
    def __init__(self, cache_dir: str = "cache/product_search"):
        """Initialize the enhanced product searcher."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_search_cache()
        
    def _init_search_cache(self):
        """Initialize the search result cache."""
        self.cache_file = self.cache_dir / "search_cache.json"
        self.cache = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, query: str, search_method: str, image_hash: Optional[str] = None) -> str:
        """Generate a cache key for the search."""
        key_parts = [query, search_method]
        if image_hash:
            key_parts.append(image_hash)
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def _extract_smart_keywords(self, caption_data: Dict[str, Any], object_class: str) -> List[str]:
        """Extract smart keywords from caption data for better search queries."""
        keywords = []
        
        # Extract style keywords
        style = caption_data.get('style', '').lower()
        if style and style != 'unknown':
            # Map common styles to search-friendly terms
            style_mapping = {
                'mid-century': 'mid-century modern',
                'contemporary': 'contemporary modern',
                'industrial': 'industrial style',
                'farmhouse': 'rustic farmhouse',
                'scandinavian': 'scandinavian minimalist',
                'traditional': 'traditional classic'
            }
            keywords.append(style_mapping.get(style, style))
        
        # Extract material keywords
        material = caption_data.get('material', '').lower()
        if material and material != 'unknown':
            # Enhance material descriptions
            material_mapping = {
                'wood': 'solid wood',
                'leather': 'genuine leather',
                'fabric': 'upholstered fabric',
                'metal': 'metal frame',
                'glass': 'tempered glass'
            }
            keywords.append(material_mapping.get(material, material))
        
        # Extract color keywords
        color = caption_data.get('colour', '').lower()
        if color and color != 'unknown':
            keywords.append(color)
        
        # Add object-specific keywords based on class
        class_keywords = {
            'chair': ['accent chair', 'armchair', 'dining chair'],
            'sofa': ['sectional sofa', 'loveseat', 'couch'],
            'table': ['coffee table', 'dining table', 'side table'],
            'bed': ['platform bed', 'upholstered bed', 'bed frame'],
            'cabinet': ['storage cabinet', 'display cabinet', 'sideboard']
        }
        
        if object_class in class_keywords:
            # Add the most relevant class keyword based on other attributes
            if 'dining' in ' '.join(keywords):
                keywords.extend([k for k in class_keywords[object_class] if 'dining' in k])
            elif 'living' in ' '.join(keywords) or 'accent' in ' '.join(keywords):
                keywords.extend([k for k in class_keywords[object_class] if 'accent' in k or 'coffee' in k])
            else:
                keywords.append(class_keywords[object_class][0])
        
        return keywords
    
    def construct_optimized_query(self, 
                                 caption_data: Dict[str, Any], 
                                 object_class: str,
                                 use_gpt_caption: bool = True) -> str:
        """Construct an optimized search query from caption data."""
        # Start with GPT-4V caption if available and enabled
        if use_gpt_caption and caption_data.get('caption'):
            base_query = caption_data['caption']
            # Remove generic terms that don't help with search
            generic_terms = ['appears to be', 'seems like', 'probably', 'possibly', 'unknown']
            for term in generic_terms:
                base_query = base_query.replace(term, '')
        else:
            base_query = object_class
        
        # Extract smart keywords
        keywords = self._extract_smart_keywords(caption_data, object_class)
        
        # Combine base query with keywords, avoiding duplicates
        query_parts = [base_query]
        for keyword in keywords:
            if keyword.lower() not in base_query.lower():
                query_parts.append(keyword)
        
        # Clean and format the final query
        final_query = ' '.join(query_parts)
        final_query = ' '.join(final_query.split())  # Remove extra spaces
        
        # Limit query length for API compatibility
        if len(final_query) > 100:
            # Prioritize style, material, and object type
            priority_parts = []
            if keywords:
                priority_parts.extend(keywords[:3])
            priority_parts.append(object_class)
            final_query = ' '.join(priority_parts)
        
        logger.info(f"Constructed optimized query: '{final_query}'")
        return final_query
    
    def rank_products_by_similarity(self,
                                   products: List[Dict[str, Any]],
                                   query_embedding: Optional[np.ndarray] = None,
                                   caption_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Rank products by visual and textual similarity."""
        if not products:
            return products
        
        # If we have embeddings, we could compute similarity scores
        # For now, apply basic ranking heuristics
        
        scored_products = []
        for product in products:
            score = 0.0
            
            # Check title relevance
            title = product.get('title', '').lower()
            if caption_data:
                # Boost score for matching style
                if caption_data.get('style', '').lower() in title:
                    score += 2.0
                # Boost score for matching material
                if caption_data.get('material', '').lower() in title:
                    score += 1.5
                # Boost score for matching color
                if caption_data.get('colour', '').lower() in title:
                    score += 1.0
            
            # Add price-based scoring (prefer mid-range prices)
            try:
                price_str = product.get('price', '')
                if '$' in price_str:
                    price = float(price_str.replace('$', '').replace(',', ''))
                    # Prefer products in reasonable price range
                    if 100 <= price <= 1000:
                        score += 0.5
                    elif price > 2000:
                        score -= 0.5
            except:
                pass
            
            scored_products.append((score, product))
        
        # Sort by score (descending)
        scored_products.sort(key=lambda x: x[0], reverse=True)
        
        # Return ranked products
        return [product for _, product in scored_products]
    
    def search_with_caching(self,
                           query: str,
                           search_method: str,
                           search_function,
                           cache_ttl: int = 3600,
                           **kwargs) -> Any:
        """Execute search with caching support."""
        # Generate cache key
        image_hash = None
        if 'image_path' in kwargs and kwargs['image_path']:
            # Create hash of image for cache key
            try:
                with open(kwargs['image_path'], 'rb') as f:
                    image_hash = hashlib.md5(f.read()).hexdigest()
            except:
                pass
        
        cache_key = self._get_cache_key(query, search_method, image_hash)
        
        # Check cache
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result['timestamp'] < cache_ttl:
                print(f"💾 Cache HIT for query: '{query[:50]}...'")
                logger.info(f"Returning cached result for query: '{query}'")
                return cached_result['data']
        
        # Execute search
        print(f"🔍 Cache MISS - executing search for: '{query[:50]}...'")
        result = search_function(query=query, **kwargs)
        print(f"✅ Search completed - found {len(result) if isinstance(result, list) else 'N/A'} results")
        
        # Cache result
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
        self._save_cache()
        
        return result
    
    def parallel_search_objects(self,
                              objects: List[Dict[str, Any]],
                              search_function,
                              max_workers: int = 3) -> List[Tuple[int, Any]]:
        """Search for products for multiple objects in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all search tasks
            future_to_idx = {}
            for idx, obj in enumerate(objects):
                # Construct optimized query for each object
                query = self.construct_optimized_query(
                    obj.get('caption_data', {}),
                    obj.get('class', 'furniture')
                )
                
                future = executor.submit(
                    search_function,
                    query=query,
                    style_info=obj.get('caption_data', {}),
                    query_embedding=obj.get('clip_embedding'),
                    image_path=obj.get('crop_path')
                )
                future_to_idx[future] = idx
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error searching for object {idx}: {e}")
                    results.append((idx, []))
        
        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        return results


def create_enhanced_search_pipeline(objects: List[Dict[str, Any]],
                                  search_method: str = 'hybrid',
                                  use_caching: bool = True) -> List[Dict[str, Any]]:
    """
    Create an enhanced search pipeline for multiple objects.
    
    Args:
        objects: List of detected objects with features
        search_method: 'text_only', 'visual_only', or 'hybrid'
        use_caching: Whether to use result caching
        
    Returns:
        List of search results for each object
    """
    print(f"🔧 Enhanced Search Pipeline: Starting for {len(objects)} objects")
    print(f"📊 Search method: {search_method}, Caching: {use_caching}")
    
    from new_product_matcher import search_products_enhanced
    
    # Initialize enhanced searcher
    searcher = EnhancedProductSearcher()
    
    # Process each object
    enhanced_results = []
    
    # Process in batches to avoid overwhelming the API
    batch_size = 5
    
    for idx, obj in enumerate(objects):
        print(f"🔍 Processing object {idx+1}/{len(objects)}: {obj.get('class', 'unknown')}")
        
        # Add small delay between API calls to avoid rate limiting
        if idx > 0 and idx % batch_size == 0:
            print(f"⏸️ Pausing briefly after {idx} objects to avoid rate limiting...")
            import time
            time.sleep(1)
        
        try:
            # Construct optimized query
            optimized_query = searcher.construct_optimized_query(
                obj.get('caption_data', {}),
                obj.get('class', 'furniture'),
                use_gpt_caption=True
            )
            print(f"📝 Generated query: '{optimized_query[:60]}...')")
            
            # Use the enhanced search from new_product_matcher with all its features
            if search_method == 'hybrid' and obj.get('crop_path'):
                # For hybrid search, use the powerful search_products_hybrid function
                from new_product_matcher import search_products_hybrid, SERP_API_KEY
                
                # Use hybrid search that combines visual and text
                crop_path = obj.get('crop_path')
                if crop_path:  # Type guard for linter
                    products = search_products_hybrid(
                        image_path=crop_path,
                        caption_data=obj.get('caption_data', {}),
                        serp_api_key_text=SERP_API_KEY,
                        serp_api_key_visual=SERP_API_KEY  # Same key for both
                    )
                else:
                    products = []
            else:
                # Use the enhanced search with caching
                if use_caching:
                    products = searcher.search_with_caching(
                        query=optimized_query,
                        search_method=search_method,
                        search_function=search_products_enhanced,
                        style_info=obj.get('caption_data', {}),
                        query_embedding=obj.get('clip_embedding'),
                        image_path=obj.get('crop_path'),
                        use_function_calling=True  # Use the better function calling agent
                    )
                else:
                    products = search_products_enhanced(
                        query=optimized_query,
                        style_info=obj.get('caption_data', {}),
                        query_embedding=obj.get('clip_embedding'),
                        image_path=obj.get('crop_path'),
                        search_method=search_method,
                        use_function_calling=True  # Use the better function calling agent
                    )
            
            # Rank products by similarity
            if isinstance(products, list):
                ranked_products = searcher.rank_products_by_similarity(
                    products,
                    query_embedding=obj.get('clip_embedding'),
                    caption_data=obj.get('caption_data', {})
                )
            else:
                ranked_products = products
            
            enhanced_results.append({
                'object_id': obj.get('id', idx),
                'object_class': obj.get('class', 'unknown'),
                'optimized_query': optimized_query,
                'products': ranked_products,
                'search_method': search_method
            })
            
        except Exception as e:
            logger.error(f"Error in enhanced search for object {idx}: {e}")
            enhanced_results.append({
                'object_id': obj.get('id', idx),
                'error': str(e),
                'products': []
            })
    
    return enhanced_results 