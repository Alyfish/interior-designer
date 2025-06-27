"""
Object detection utilities for the interior designer app.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

def detect_room_objects(image, model_path: str = None) -> List[Dict[str, Any]]:
    """
    Detect objects in a room image using YOLOv8.
    
    Args:
        image: PIL Image or path to image
        model_path: Path to YOLOv8 model (optional)
    
    Returns:
        List of detected objects with bounding boxes and confidence scores
    """
    try:
        from new_object_detector import ObjectDetector
        
        detector = ObjectDetector(model_path=model_path)
        return detector.detect(image)
        
    except Exception as e:
        logger.error(f"Error in detect_room_objects: {e}")
        return []

def get_furniture_objects(detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter detected objects to only return furniture items.
    
    Args:
        detected_objects: List of all detected objects
    
    Returns:
        List of furniture objects only
    """
    furniture_classes = {
        'chair', 'couch', 'sofa', 'bed', 'dining table', 'table', 
        'desk', 'tv', 'laptop', 'keyboard', 'mouse', 'remote', 
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
        'book', 'clock', 'vase', 'scissors', 'hair drier', 'toothbrush'
    }
    
    furniture_objects = []
    for obj in detected_objects:
        if obj.get('class_name', '').lower() in furniture_classes:
            furniture_objects.append(obj)
    
    return furniture_objects

def crop_object_from_image(image: Image.Image, bbox: Tuple[int, int, int, int], 
                          padding: int = 10) -> Image.Image:
    """
    Crop an object from an image using its bounding box.
    
    Args:
        image: PIL Image
        bbox: Bounding box as (x1, y1, x2, y2)
        padding: Additional padding around the object
    
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    
    return image.crop((x1, y1, x2, y2))

def enhance_object_detection(image: Image.Image, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Enhanced object detection with post-processing.
    
    Args:
        image: PIL Image to detect objects in
        confidence_threshold: Minimum confidence for object detection
    
    Returns:
        List of enhanced detected objects
    """
    # Use the main object detector
    objects = detect_room_objects(image)
    
    # Filter by confidence
    enhanced_objects = []
    for obj in objects:
        if obj.get('confidence', 0) >= confidence_threshold:
            enhanced_objects.append(obj)
    
    # Sort by confidence (highest first)
    enhanced_objects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    return enhanced_objects 