"""
Smart Product Search Agent

This module provides intelligent product search capabilities that enhance
the existing search functionality without modifying it. All features are
additive and can be disabled via feature flags.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import existing functionality - DO NOT MODIFY THESE
from new_product_matcher import (
    search_products_serpapi_tool,
    search_products_reverse_image_serpapi,
    search_products_hybrid,
    parse_agent_response_to_products,
    SERP_API_KEY
)
from config import (
    USE_SMART_PRODUCT_SEARCH,
    SMART_SEARCH_TIMEOUT,
    SMART_SEARCH_MAX_RESULTS,
    SMART_SEARCH_PRICE_ANALYSIS,
    SMART_SEARCH_STYLE_MATCHING,
    SMART_SEARCH_DEBUG
)

logger = logging.getLogger(__name__)


class SmartProductSearchAgent:
    """
    Intelligent product search agent that enhances existing search capabilities
    without modifying core functionality.
    """
    
    def __init__(self):
        """Initialize the smart search agent."""
        self.serp_api_key = SERP_API_KEY
        self.timeout = SMART_SEARCH_TIMEOUT
        self.max_results = SMART_SEARCH_MAX_RESULTS
        self.enable_price_analysis = SMART_SEARCH_PRICE_ANALYSIS
        self.enable_style_matching = SMART_SEARCH_STYLE_MATCHING
        self.debug = SMART_SEARCH_DEBUG
        
        if self.debug:
            logger.info("SmartProductSearchAgent initialized with settings:")
            logger.info(f"  - Timeout: {self.timeout}s")
            logger.info(f"  - Max results: {self.max_results}")
            logger.info(f"  - Price analysis: {self.enable_price_analysis}")
            logger.info(f"  - Style matching: {self.enable_style_matching}")
    
    def analyze_room_context(self, caption_data: Dict[str, Any], object_class: str) -> Dict[str, Any]:
        """
        Analyze the room context from caption data to make intelligent search decisions.
        
        Args:
            caption_data: Caption data from vision analysis
            object_class: The class of the detected object
            
        Returns:
            Context dictionary with room analysis
        """
        context = {
            'object_class': object_class,
            'style': caption_data.get('style', 'unknown').lower(),
            'material': caption_data.get('material', 'unknown').lower(),
            'color': caption_data.get('colour', 'unknown').lower(),
            'era': caption_data.get('era', 'unknown').lower(),
            'caption': caption_data.get('caption', ''),
            'quality_indicators': [],
            'estimated_price_range': None,
            'room_type': None
        }
        
        # Analyze quality indicators from caption
        caption_lower = context['caption'].lower()
        quality_keywords = {
            'luxury': ['luxury', 'premium', 'high-end', 'designer', 'elegant'],
            'mid-range': ['modern', 'contemporary', 'stylish', 'quality'],
            'budget': ['simple', 'basic', 'minimal', 'affordable']
        }
        
        for quality_level, keywords in quality_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                context['quality_indicators'].append(quality_level)
        
        # Estimate price range based on quality and object type
        if self.enable_price_analysis:
            context['estimated_price_range'] = self._estimate_price_range(
                object_class, context['quality_indicators'], context['material']
            )
        
        # Determine room type from caption
        room_keywords = {
            'living_room': ['living', 'lounge', 'family room', 'sitting'],
            'bedroom': ['bedroom', 'sleeping', 'bed'],
            'dining': ['dining', 'eating', 'kitchen'],
            'office': ['office', 'study', 'work', 'desk']
        }
        
        for room_type, keywords in room_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                context['room_type'] = room_type
                break
        
        if self.debug:
            logger.info(f"Room context analysis: {json.dumps(context, indent=2)}")
        
        return context
    
    def _estimate_price_range(self, object_class: str, quality_indicators: List[str], material: str) -> Tuple[float, float]:
        """
        Estimate appropriate price range based on object type and quality indicators.
        
        Args:
            object_class: Type of furniture
            quality_indicators: List of quality levels detected
            material: Material of the item
            
        Returns:
            Tuple of (min_price, max_price)
        """
        # Base price ranges by object type
        base_ranges = {
            'chair': (50, 500),
            'sofa': (200, 2000),
            'table': (100, 1500),
            'bed': (300, 3000),
            'cabinet': (150, 2000),
            'lamp': (30, 300),
            'rug': (50, 500),
            'mirror': (30, 300),
            'ottoman': (50, 400),
            'desk': (100, 1000)
        }
        
        # Get base range
        min_price, max_price = base_ranges.get(object_class.lower(), (50, 500))
        
        # Adjust based on quality indicators
        if 'luxury' in quality_indicators:
            min_price *= 2
            max_price *= 3
        elif 'budget' in quality_indicators:
            min_price *= 0.5
            max_price *= 0.7
        
        # Adjust based on material
        premium_materials = ['leather', 'solid wood', 'marble', 'brass', 'velvet']
        if any(mat in material.lower() for mat in premium_materials):
            min_price *= 1.5
            max_price *= 1.5
        
        return (round(min_price), round(max_price))
    
    def generate_smart_queries(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple intelligent search queries based on context.
        
        Args:
            context: Room context analysis
            
        Returns:
            List of query dictionaries with strategy information
        """
        queries = []
        base_object = context['object_class']
        
        # Query 1: Style-focused search
        if context['style'] != 'unknown' and self.enable_style_matching:
            style_query = f"{context['style']} {base_object}"
            if context['material'] != 'unknown':
                style_query += f" {context['material']}"
            queries.append({
                'query': style_query,
                'strategy': 'style_match',
                'priority': 1
            })
        
        # Query 2: Price-targeted search
        if context['estimated_price_range'] and self.enable_price_analysis:
            min_price, max_price = context['estimated_price_range']
            price_query = f"{base_object} ${min_price}-${max_price}"
            if context['color'] != 'unknown':
                price_query = f"{context['color']} {price_query}"
            queries.append({
                'query': price_query,
                'strategy': 'price_targeted',
                'priority': 2
            })
        
        # Query 3: Feature-based search
        feature_query = base_object
        features = []
        if context['room_type']:
            features.append(f"for {context['room_type']}")
        if context['quality_indicators']:
            features.append(context['quality_indicators'][0])
        if features:
            feature_query += " " + " ".join(features)
        queries.append({
            'query': feature_query,
            'strategy': 'feature_based',
            'priority': 3
        })
        
        # Query 4: Alternative terminology
        alternative_terms = {
            'sofa': ['couch', 'settee', 'loveseat'],
            'chair': ['armchair', 'accent chair', 'lounge chair'],
            'table': ['desk', 'console', 'stand'],
            'cabinet': ['storage unit', 'sideboard', 'credenza']
        }
        
        if base_object.lower() in alternative_terms:
            for alt_term in alternative_terms[base_object.lower()][:2]:
                alt_query = context['caption'].replace(base_object, alt_term)
                queries.append({
                    'query': alt_query,
                    'strategy': 'alternative_terms',
                    'priority': 4
                })
        
        if self.debug:
            logger.info(f"Generated {len(queries)} smart queries:")
            for q in queries:
                logger.info(f"  - [{q['strategy']}] {q['query']}")
        
        return queries
    
    def execute_parallel_searches(self, queries: List[Dict[str, Any]], image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute multiple searches in parallel using existing search functions.
        
        Args:
            queries: List of query dictionaries
            image_path: Optional image path for visual search
            
        Returns:
            List of all search results
        """
        all_results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit search tasks
            future_to_query = {}
            
            for query_info in queries[:4]:  # Limit parallel searches
                # Use existing search function - DO NOT MODIFY
                future = executor.submit(
                    search_products_serpapi_tool,
                    query=query_info['query'],
                    serp_api_key=self.serp_api_key
                )
                future_to_query[future] = query_info
            
            # Collect results as they complete
            for future in as_completed(future_to_query, timeout=self.timeout):
                query_info = future_to_query[future]
                try:
                    result = future.result()
                    # Parse using existing parser - DO NOT MODIFY
                    products = parse_agent_response_to_products(result)
                    
                    # Add metadata to products
                    for product in products:
                        product['search_strategy'] = query_info['strategy']
                        product['search_priority'] = query_info['priority']
                    
                    all_results.extend(products)
                    
                except Exception as e:
                    logger.error(f"Search failed for query '{query_info['query']}': {e}")
        
        return all_results
    
    def score_products(self, products: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Score and rank products based on multiple criteria.
        
        Args:
            products: List of product dictionaries
            context: Room context analysis
            
        Returns:
            List of scored and sorted products
        """
        scored_products = []
        
        for product in products:
            score = 0.0
            score_breakdown = {}
            
            title_lower = product.get('title', '').lower()
            
            # Style matching score
            if self.enable_style_matching and context['style'] != 'unknown':
                if context['style'] in title_lower:
                    score += 2.0
                    score_breakdown['style_match'] = 2.0
            
            # Material matching score
            if context['material'] != 'unknown':
                if context['material'] in title_lower:
                    score += 1.5
                    score_breakdown['material_match'] = 1.5
            
            # Color matching score
            if context['color'] != 'unknown':
                if context['color'] in title_lower:
                    score += 1.0
                    score_breakdown['color_match'] = 1.0
            
            # Price appropriateness score
            if self.enable_price_analysis and context['estimated_price_range']:
                try:
                    price_str = product.get('price', '')
                    if '$' in price_str:
                        price = float(price_str.replace('$', '').replace(',', '').split('-')[0])
                        min_price, max_price = context['estimated_price_range']
                        
                        if min_price <= price <= max_price:
                            score += 1.5
                            score_breakdown['price_appropriate'] = 1.5
                        elif price < min_price * 0.5 or price > max_price * 2:
                            score -= 1.0
                            score_breakdown['price_inappropriate'] = -1.0
                except:
                    pass
            
            # Search strategy bonus
            strategy_scores = {
                'style_match': 0.5,
                'price_targeted': 0.3,
                'feature_based': 0.2,
                'alternative_terms': 0.1
            }
            strategy = product.get('search_strategy', '')
            if strategy in strategy_scores:
                score += strategy_scores[strategy]
                score_breakdown['strategy_bonus'] = strategy_scores[strategy]
            
            # Retailer quality score (simplified for now)
            retailer = product.get('retailer', '').lower()
            quality_retailers = ['wayfair', 'west elm', 'cb2', 'crate and barrel', 'pottery barn']
            if any(r in retailer for r in quality_retailers):
                score += 0.5
                score_breakdown['quality_retailer'] = 0.5
            
            product['smart_score'] = score
            product['score_breakdown'] = score_breakdown
            scored_products.append(product)
        
        # Sort by score (descending)
        scored_products.sort(key=lambda x: x['smart_score'], reverse=True)
        
        if self.debug:
            logger.info(f"Scored {len(scored_products)} products")
            for i, p in enumerate(scored_products[:5]):
                logger.info(f"  {i+1}. {p['title'][:50]}... Score: {p['smart_score']:.2f}")
        
        return scored_products
    
    def deduplicate_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate products based on title similarity.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Deduplicated list of products
        """
        seen_titles = set()
        unique_products = []
        
        for product in products:
            # Create a normalized title for comparison
            title = product.get('title', '').lower()
            # Remove common words that don't affect uniqueness
            for word in ['the', 'a', 'an', 'by', 'from', 'at']:
                title = title.replace(f' {word} ', ' ')
            title_key = ''.join(title.split()[:5])  # Use first 5 words
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_products.append(product)
        
        return unique_products
    
    def search_products_intelligently(self,
                                    caption_data: Dict[str, Any],
                                    object_class: str,
                                    image_path: Optional[str] = None,
                                    clip_embedding: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Main entry point for intelligent product search.
        
        Args:
            caption_data: Caption data from vision analysis
            object_class: Class of the detected object
            image_path: Optional path to image for visual search
            clip_embedding: Optional CLIP embedding (not used yet)
            
        Returns:
            List of intelligently selected and ranked products
        """
        if not USE_SMART_PRODUCT_SEARCH:
            logger.warning("Smart search is disabled via feature flag")
            return []
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze room context
            context = self.analyze_room_context(caption_data, object_class)
            
            # Step 2: Generate smart queries
            queries = self.generate_smart_queries(context)
            
            # Step 3: Execute searches in parallel
            all_products = self.execute_parallel_searches(queries, image_path)
            
            # Step 4: Deduplicate results
            unique_products = self.deduplicate_products(all_products)
            
            # Step 5: Score and rank products
            scored_products = self.score_products(unique_products, context)
            
            # Step 6: Limit results
            final_products = scored_products[:self.max_results]
            
            # Add metadata
            for i, product in enumerate(final_products):
                product['rank'] = i + 1
                product['search_time'] = time.time() - start_time
                product['search_method'] = 'smart_search'
            
            if self.debug:
                logger.info(f"Smart search completed in {time.time() - start_time:.2f}s")
                logger.info(f"Found {len(final_products)} products from {len(all_products)} total results")
            
            return final_products
            
        except Exception as e:
            logger.error(f"Smart search failed: {e}")
            # Gracefully fall back to empty results
            return []