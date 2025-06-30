"""
Style Compatibility Scorer - Enhances product recommendations
Works alongside existing search without breaking it
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class StyleCompatibilityScorer:
    """Scores products based on style compatibility with room context"""
    
    def __init__(self):
        # Style compatibility matrix (simplified for MVP)
        self.style_compatibility = {
            'modern': {
                'modern': 1.0, 'contemporary': 0.9, 'minimalist': 0.8,
                'industrial': 0.6, 'traditional': 0.3, 'vintage': 0.2
            },
            'traditional': {
                'traditional': 1.0, 'classic': 0.9, 'vintage': 0.8,
                'rustic': 0.6, 'modern': 0.3, 'industrial': 0.2
            },
            'scandinavian': {
                'scandinavian': 1.0, 'minimalist': 0.9, 'modern': 0.8,
                'contemporary': 0.7, 'traditional': 0.4, 'industrial': 0.5
            },
            'industrial': {
                'industrial': 1.0, 'modern': 0.7, 'rustic': 0.8,
                'vintage': 0.6, 'traditional': 0.3, 'scandinavian': 0.5
            },
            'bohemian': {
                'bohemian': 1.0, 'eclectic': 0.9, 'vintage': 0.7,
                'rustic': 0.6, 'modern': 0.4, 'industrial': 0.5
            }
        }
        
        # Color harmony rules
        self.color_harmony_boost = 0.2
    
    def score_products(self, 
                      products: List[Dict], 
                      room_context: Dict,
                      selected_object: Dict) -> List[Dict]:
        """
        Add compatibility scores to existing products
        
        Args:
            products: List of products from search
            room_context: Room analysis results
            selected_object: The object being matched
            
        Returns:
            Products with added 'style_score' and 'context_score'
        """
        if not products:
            return products
            
        scored_products = []
        
        for product in products:
            try:
                # Start with existing product data
                scored_product = product.copy()
                
                # Calculate style compatibility
                style_score = self._calculate_style_score(
                    product, 
                    room_context.get('detected_styles', []),
                    selected_object.get('caption_data', {})
                )
                
                # Calculate color harmony
                color_score = self._calculate_color_harmony(
                    product,
                    room_context.get('dominant_colors', []),
                    selected_object
                )
                
                # Calculate size appropriateness
                size_score = self._calculate_size_score(
                    product,
                    room_context,
                    selected_object
                )
                
                # Combine scores (weighted average)
                context_score = (
                    0.4 * style_score +
                    0.3 * color_score +
                    0.3 * size_score
                )
                
                # Add scores to product
                scored_product['style_score'] = float(style_score)
                scored_product['color_score'] = float(color_score)
                scored_product['size_score'] = float(size_score)
                scored_product['context_score'] = float(context_score)
                
                # Boost original search score if available
                if 'score' in scored_product:
                    scored_product['combined_score'] = (
                        0.7 * scored_product['score'] + 
                        0.3 * context_score
                    )
                else:
                    scored_product['combined_score'] = context_score
                
                scored_products.append(scored_product)
                
            except Exception as e:
                logger.warning(f"Failed to score product: {e}")
                # Add product unchanged if scoring fails
                scored_products.append(product)
        
        # Sort by combined score
        try:
            scored_products.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        except Exception as e:
            logger.warning(f"Failed to sort products: {e}")
        
        return scored_products
    
    def _calculate_style_score(self, product: Dict, room_styles: List[str], object_data: Dict) -> float:
        """Calculate style compatibility score"""
        try:
            # Extract style from product title/description
            product_text = f"{product.get('title', '')} {product.get('description', '')}".lower()
            
            # Simple keyword matching for MVP
            detected_style = None
            style_keywords = {
                'modern': ['modern', 'contemporary', 'sleek', 'minimal'],
                'traditional': ['traditional', 'classic', 'ornate', 'vintage'],
                'industrial': ['industrial', 'metal', 'rustic', 'raw'],
                'scandinavian': ['scandinavian', 'nordic', 'swedish', 'danish'],
                'bohemian': ['bohemian', 'boho', 'eclectic', 'artistic']
            }
            
            for style, keywords in style_keywords.items():
                if any(keyword in product_text for keyword in keywords):
                    detected_style = style
                    break
            
            if not detected_style:
                return 0.5  # Neutral score if style unknown
            
            # Check compatibility
            max_score = 0.5
            
            # First check object's own style
            object_style = object_data.get('style', '').lower()
            if object_style in self.style_compatibility:
                score = self.style_compatibility[object_style].get(detected_style, 0.5)
                max_score = max(max_score, score)
            
            # Then check room styles
            for room_style in room_styles:
                if room_style in self.style_compatibility:
                    score = self.style_compatibility[room_style].get(detected_style, 0.5)
                    max_score = max(max_score, score)
            
            return max_score
            
        except Exception as e:
            logger.warning(f"Style score calculation failed: {e}")
            return 0.5
    
    def _calculate_color_harmony(self, product: Dict, room_colors: List[tuple], object: Dict) -> float:
        """Calculate color harmony score"""
        try:
            # For MVP, use simple heuristics
            product_title = product.get('title', '').lower()
            
            # Extract product color
            color_keywords = ['black', 'white', 'gray', 'grey', 'brown', 'beige', 
                             'blue', 'green', 'red', 'yellow', 'orange', 'purple',
                             'navy', 'teal', 'pink', 'ivory', 'cream', 'charcoal']
            
            product_colors = [color for color in color_keywords if color in product_title]
            
            if not product_colors:
                return 0.6  # Neutral score
            
            # Simple harmony rules
            harmony_score = 0.5
            
            # Neutral colors go with everything
            neutral_colors = ['black', 'white', 'gray', 'grey', 'beige', 'ivory', 'cream', 'charcoal']
            if any(color in neutral_colors for color in product_colors):
                harmony_score = 0.8
            
            # Boost if matches room's dominant colors (simplified)
            # In production, would use proper color distance calculations
            if room_colors and len(room_colors) > 0:
                # Check if product color is similar to room colors
                # This is simplified - real implementation would use LAB color space
                harmony_score += self.color_harmony_boost
            
            return min(harmony_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Color harmony calculation failed: {e}")
            return 0.6
    
    def _calculate_size_score(self, product: Dict, room_context: Dict, object: Dict) -> float:
        """Calculate size appropriateness score"""
        try:
            # Simple heuristics based on room density
            density = room_context.get('object_density', 5)
            
            if density < 3:  # Sparse room
                return 0.9  # Most items appropriate
            elif density < 7:  # Normal density
                return 0.8
            else:  # Crowded room
                # Prefer smaller items
                product_text = product.get('title', '').lower()
                if any(word in product_text for word in ['compact', 'small', 'mini', 'slim', 'space-saving']):
                    return 0.9
                elif any(word in product_text for word in ['large', 'oversized', 'big', 'grand']):
                    return 0.4
                return 0.7
                
        except Exception as e:
            logger.warning(f"Size score calculation failed: {e}")
            return 0.7