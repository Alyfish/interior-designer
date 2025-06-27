"""
Image analysis utilities for room style, features, and dimensions.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)

def analyze_room_style(image: Image.Image, detected_objects: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze the style of a room from an image.
    
    Args:
        image: PIL Image of the room
        detected_objects: Optional list of detected objects to help with analysis
    
    Returns:
        Dictionary containing style analysis
    """
    try:
        # Use vision features for advanced analysis
        from vision_features import get_caption_json
        
        # Get detailed room analysis
        caption_data = get_caption_json(image)
        
        # Extract style information
        style_info = {
            'primary_style': 'Modern',  # Default
            'color_scheme': 'Neutral',
            'materials': [],
            'era': 'Contemporary',
            'formality': 'Casual',
            'confidence': 0.7
        }
        
        # Parse caption data for style indicators
        if caption_data and isinstance(caption_data, dict):
            room_analysis = caption_data.get('room_analysis', {})
            style_info.update({
                'primary_style': room_analysis.get('style', 'Modern'),
                'color_scheme': room_analysis.get('color_scheme', 'Neutral'),
                'materials': room_analysis.get('materials', []),
                'formality': room_analysis.get('formality', 'Casual')
            })
        
        # Enhance with object-based style detection
        if detected_objects:
            style_info.update(_analyze_style_from_objects(detected_objects))
        
        return style_info
        
    except Exception as e:
        logger.error(f"Error in analyze_room_style: {e}")
        return {
            'primary_style': 'Modern',
            'color_scheme': 'Neutral', 
            'materials': ['wood', 'fabric'],
            'era': 'Contemporary',
            'formality': 'Casual',
            'confidence': 0.5
        }

def extract_room_features(image: Image.Image, detected_objects: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract key features from a room image.
    
    Args:
        image: PIL Image of the room
        detected_objects: Optional list of detected objects
    
    Returns:
        Dictionary containing room features
    """
    features = {
        'lighting': 'Natural',
        'layout': 'Open',
        'focal_points': [],
        'color_palette': ['white', 'gray', 'beige'],
        'textures': ['smooth', 'soft'],
        'architectural_elements': []
    }
    
    try:
        # Analyze color palette
        features['color_palette'] = _extract_color_palette(image)
        
        # Analyze lighting
        features['lighting'] = _analyze_lighting(image)
        
        # Extract focal points from detected objects
        if detected_objects:
            features['focal_points'] = _identify_focal_points(detected_objects)
            features['layout'] = _analyze_layout(detected_objects)
        
        return features
        
    except Exception as e:
        logger.error(f"Error in extract_room_features: {e}")
        return features

def estimate_room_dimensions(image: Image.Image, detected_objects: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Estimate room dimensions and scale.
    
    Args:
        image: PIL Image of the room
        detected_objects: List of detected objects for scale reference
    
    Returns:
        Dictionary containing dimension estimates
    """
    dimensions = {
        'estimated_width': 12.0,  # feet
        'estimated_length': 15.0,  # feet
        'estimated_height': 9.0,   # feet
        'scale_confidence': 0.3,
        'reference_objects': []
    }
    
    try:
        if detected_objects:
            # Use furniture objects as scale references
            scale_references = _get_scale_references(detected_objects)
            if scale_references:
                dimensions.update(_estimate_from_references(image, scale_references))
                dimensions['reference_objects'] = [obj['class_name'] for obj in scale_references]
                dimensions['scale_confidence'] = 0.6
        
        return dimensions
        
    except Exception as e:
        logger.error(f"Error in estimate_room_dimensions: {e}")
        return dimensions

def get_room_type_from_objects(detected_objects: List[Dict[str, Any]]) -> str:
    """
    Determine room type based on detected objects.
    
    Args:
        detected_objects: List of detected objects
    
    Returns:
        Room type string
    """
    if not detected_objects:
        return 'Living Room'
    
    # Room type indicators
    bedroom_objects = {'bed', 'nightstand', 'dresser', 'wardrobe'}
    kitchen_objects = {'refrigerator', 'microwave', 'oven', 'sink', 'stove'}
    bathroom_objects = {'toilet', 'sink', 'bathtub', 'shower'}
    dining_room_objects = {'dining table', 'chair'}
    living_room_objects = {'couch', 'sofa', 'tv', 'coffee table'}
    office_objects = {'desk', 'computer', 'office chair', 'bookshelf'}
    
    # Count object types
    object_names = {obj.get('class_name', '').lower() for obj in detected_objects}
    
    scores = {
        'Bedroom': len(object_names & bedroom_objects),
        'Kitchen': len(object_names & kitchen_objects),
        'Bathroom': len(object_names & bathroom_objects), 
        'Dining Room': len(object_names & dining_room_objects),
        'Living Room': len(object_names & living_room_objects),
        'Office': len(object_names & office_objects)
    }
    
    # Return room type with highest score
    return max(scores, key=scores.get) if any(scores.values()) else 'Living Room'

# Helper functions

def _analyze_style_from_objects(detected_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze style indicators from detected objects."""
    style_indicators = {
        'primary_style': 'Modern',
        'materials': [],
        'confidence': 0.6
    }
    
    # Simple style detection based on furniture types
    object_names = [obj.get('class_name', '').lower() for obj in detected_objects]
    
    if 'antique' in ' '.join(object_names) or 'vintage' in ' '.join(object_names):
        style_indicators['primary_style'] = 'Vintage'
    elif 'modern' in ' '.join(object_names) or 'contemporary' in ' '.join(object_names):
        style_indicators['primary_style'] = 'Modern'
    
    return style_indicators

def _extract_color_palette(image: Image.Image) -> List[str]:
    """Extract dominant colors from image."""
    try:
        # Simple color extraction - can be enhanced with clustering
        img_array = np.array(image.resize((50, 50)))
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Convert to basic color names
        r, g, b = avg_color
        if r > 200 and g > 200 and b > 200:
            return ['white', 'light gray']
        elif r < 100 and g < 100 and b < 100:
            return ['black', 'dark gray']
        elif r > g and r > b:
            return ['red', 'warm tones']
        elif g > r and g > b:
            return ['green', 'natural tones']
        elif b > r and b > g:
            return ['blue', 'cool tones']
        else:
            return ['neutral', 'beige', 'gray']
            
    except Exception as e:
        logger.error(f"Error extracting color palette: {e}")
        return ['neutral', 'white', 'gray']

def _analyze_lighting(image: Image.Image) -> str:
    """Analyze lighting conditions in the image."""
    try:
        # Simple brightness analysis
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        avg_brightness = np.mean(img_array)
        
        if avg_brightness > 180:
            return 'Bright Natural'
        elif avg_brightness > 120:
            return 'Natural'
        elif avg_brightness > 80:
            return 'Ambient'
        else:
            return 'Low Light'
            
    except Exception as e:
        logger.error(f"Error analyzing lighting: {e}")
        return 'Natural'

def _identify_focal_points(detected_objects: List[Dict[str, Any]]) -> List[str]:
    """Identify focal points based on object prominence."""
    focal_objects = ['tv', 'fireplace', 'bed', 'dining table', 'artwork']
    
    focal_points = []
    for obj in detected_objects:
        obj_name = obj.get('class_name', '').lower()
        if obj_name in focal_objects:
            focal_points.append(obj_name)
    
    return focal_points

def _analyze_layout(detected_objects: List[Dict[str, Any]]) -> str:
    """Analyze room layout based on object positions."""
    if len(detected_objects) < 3:
        return 'Minimal'
    elif len(detected_objects) > 8:
        return 'Crowded'
    else:
        return 'Balanced'

def _get_scale_references(detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get objects that can be used as scale references."""
    # Objects with known approximate sizes
    scale_objects = {
        'chair': {'width': 2.0, 'height': 3.0},  # feet
        'dining table': {'width': 6.0, 'length': 4.0},
        'bed': {'width': 6.0, 'length': 7.0},
        'tv': {'width': 4.0, 'height': 2.5}
    }
    
    references = []
    for obj in detected_objects:
        obj_name = obj.get('class_name', '').lower()
        if obj_name in scale_objects:
            obj['reference_size'] = scale_objects[obj_name]
            references.append(obj)
    
    return references

def _estimate_from_references(image: Image.Image, scale_references: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate room dimensions using scale reference objects."""
    # Simplified dimension estimation
    # In a real implementation, this would use object bounding boxes and perspective
    
    room_estimates = {
        'estimated_width': 12.0,
        'estimated_length': 15.0, 
        'estimated_height': 9.0,
        'scale_confidence': 0.7
    }
    
    # Adjust based on number and type of reference objects
    if len(scale_references) >= 2:
        room_estimates['scale_confidence'] = 0.8
    
    return room_estimates 