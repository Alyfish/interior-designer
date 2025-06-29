"""
Product Integration Module

This module bridges object detection with product matching functionality.
It processes detected objects, extracts visual features, and finds matching products
without modifying the existing object detection pipeline.
"""

import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import existing modules
from vision_features import extract_clip_embedding, get_caption_json
from new_product_matcher import ProductMatcher
from utils.hybrid_ranking import hybrid_rank_results

logger = logging.getLogger(__name__)

class ProductIntegration:
    """Integrates object detection with product matching."""
    
    def __init__(self, openai_api_key: Optional[str] = None, serp_api_key: Optional[str] = None):
        """
        Initialize product integration module.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4V captions
            serp_api_key: SerpAPI key for product search
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.serp_api_key = serp_api_key or os.getenv("SERP_API_KEY")
        
        # Initialize product matcher
        self.product_matcher = ProductMatcher()
        
        # Cache for embeddings and search results
        self.embedding_cache = {}
        self.search_cache = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def process_detected_object(self, 
                              original_image: np.ndarray,
                              obj: Dict[str, Any],
                              use_gpt4v: bool = True) -> Dict[str, Any]:
        """
        Process a single detected object to prepare for product matching.
        
        Args:
            original_image: Original image as numpy array (BGR)
            obj: Detected object dictionary with bbox, contours, class, etc.
            use_gpt4v: Whether to use GPT-4V for caption generation
            
        Returns:
            Enhanced object dictionary with visual features
        """
        try:
            # Extract object crop
            x1, y1, x2, y2 = obj['bbox']
            crop = original_image[y1:y2, x1:x2]
            
            # Generate unique cache key
            cache_key = f"{obj['id']}_{x1}_{y1}_{x2}_{y2}"
            
            # Check cache first
            if cache_key in self.embedding_cache:
                logger.info(f"Using cached features for object {obj['id']}")
                obj['clip_embedding'] = self.embedding_cache[cache_key]['embedding']
                obj['caption_data'] = self.embedding_cache[cache_key]['caption_data']
                return obj
            
            # Generate structured caption using vision_features
            caption_data = get_caption_json(crop, designer=True)
            obj['caption_data'] = caption_data
            
            # Create search query from caption data
            style = caption_data.get('style', '')
            material = caption_data.get('material', '')
            color = caption_data.get('colour', '')
            base_class = obj['class']
            
            # Build rich search query
            query_parts = []
            if style and style != 'unknown':
                query_parts.append(style)
            if material and material != 'unknown':
                query_parts.append(material)
            if color and color != 'unknown':
                query_parts.append(color)
            query_parts.append(base_class)
            
            obj['search_query'] = ' '.join(query_parts)
            obj['rich_caption'] = caption_data.get('caption', obj['search_query'])
            
            # Extract CLIP embedding for visual similarity
            # Save crop temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, crop)
                tmp_path = tmp.name
            
            try:
                embedding = extract_clip_embedding(tmp_path)
                obj['clip_embedding'] = embedding
                
                # Cache the results
                self.embedding_cache[cache_key] = {
                    'embedding': embedding,
                    'caption_data': caption_data
                }
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            logger.info(f"Processed object {obj['id']}: {obj['search_query']}")
            return obj
            
        except Exception as e:
            logger.error(f"Error processing object {obj['id']}: {e}")
            obj['search_query'] = obj['class']  # Fallback to basic class name
            obj['clip_embedding'] = None
            return obj
    
    def find_products_for_object(self,
                                obj: Dict[str, Any],
                                max_results: int = 10,
                                use_visual_similarity: bool = True) -> List[Dict[str, Any]]:
        """
        Find matching products for a detected object.
        
        Args:
            obj: Processed object dictionary with search_query and optionally clip_embedding
            max_results: Maximum number of products to return
            use_visual_similarity: Whether to use visual similarity ranking
            
        Returns:
            List of product matches with scores
        """
        try:
            # Check cache
            cache_key = f"{obj.get('search_query', '')}_{obj.get('id', '')}"
            if cache_key in self.search_cache:
                logger.info(f"Using cached search results for {cache_key}")
                return self.search_cache[cache_key][:max_results]
            
            # Search for products using the matcher
            search_query = obj.get('rich_caption') or obj.get('search_query', obj['class'])
            
            # Use the product matcher to search
            results = self.product_matcher.search_products(
                search_query,
                enable_reverse_image_search=False  # We'll do visual ranking separately
            )
            
            # Apply visual similarity ranking if embedding available
            if use_visual_similarity and obj.get('clip_embedding') is not None:
                results = hybrid_rank_results(
                    results,
                    query_embedding=obj['clip_embedding'],
                    text_weight=0.3,
                    visual_weight=0.7
                )
            
            # Cache results
            self.search_cache[cache_key] = results
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error finding products for object {obj.get('id')}: {e}")
            return []
    
    def process_all_objects(self,
                           original_image: np.ndarray,
                           detected_objects: List[Dict[str, Any]],
                           parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Process all detected objects in parallel.
        
        Args:
            original_image: Original image as numpy array
            detected_objects: List of detected objects
            parallel: Whether to process in parallel
            
        Returns:
            List of processed objects with visual features
        """
        # Filter out walls and floors
        furniture_objects = [
            obj for obj in detected_objects 
            if obj['class'] not in ['wall', 'floor', 'ceiling']
        ]
        
        logger.info(f"Processing {len(furniture_objects)} furniture objects")
        
        if parallel and len(furniture_objects) > 1:
            # Process in parallel
            futures = []
            for obj in furniture_objects:
                future = self.executor.submit(
                    self.process_detected_object,
                    original_image,
                    obj
                )
                futures.append(future)
            
            # Collect results
            processed_objects = []
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    processed_objects.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
        else:
            # Process sequentially
            processed_objects = [
                self.process_detected_object(original_image, obj)
                for obj in furniture_objects
            ]
        
        return processed_objects
    
    def get_product_matches_for_scene(self,
                                     original_image: np.ndarray,
                                     detected_objects: List[Dict[str, Any]],
                                     max_products_per_object: int = 5) -> Dict[str, Any]:
        """
        Get product matches for all detected objects in a scene.
        
        Args:
            original_image: Original image
            detected_objects: List of detected objects
            max_products_per_object: Max products to return per object
            
        Returns:
            Dictionary mapping object IDs to product matches
        """
        # Process all objects to extract features
        processed_objects = self.process_all_objects(original_image, detected_objects)
        
        # Find products for each object
        results = {}
        for obj in processed_objects:
            obj_id = obj['id']
            products = self.find_products_for_object(
                obj,
                max_results=max_products_per_object
            )
            
            results[obj_id] = {
                'object': obj,
                'products': products,
                'search_query': obj.get('search_query', obj['class']),
                'caption_data': obj.get('caption_data', {})
            }
            
        logger.info(f"Found products for {len(results)} objects")
        return results
    
    def format_product_display(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format product data for display in UI.
        
        Args:
            product: Raw product data
            
        Returns:
            Formatted product dictionary
        """
        return {
            'title': product.get('title', 'Unknown Product'),
            'price': product.get('price', 'N/A'),
            'source': product.get('source', 'Unknown'),
            'link': product.get('link', '#'),
            'thumbnail': product.get('thumbnail') or product.get('image'),
            'visual_score': product.get('visual_score', 0.0),
            'hybrid_score': product.get('hybrid_score', 0.0),
            'in_stock': product.get('in_stock', True)
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.search_cache.clear()
        logger.info("Cleared product integration caches")


# Convenience functions for Streamlit integration
_integration_instance = None

def get_product_integration():
    """Get or create singleton instance of ProductIntegration."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = ProductIntegration()
    return _integration_instance

def process_and_match_products(original_image: np.ndarray,
                             detected_objects: List[Dict[str, Any]],
                             max_products: int = 5) -> Dict[str, Any]:
    """
    Convenience function for Streamlit to process objects and find products.
    
    Args:
        original_image: Original image array
        detected_objects: List of detected objects
        max_products: Maximum products per object
        
    Returns:
        Product matches for all objects
    """
    integration = get_product_integration()
    return integration.get_product_matches_for_scene(
        original_image,
        detected_objects,
        max_products_per_object=max_products
    )