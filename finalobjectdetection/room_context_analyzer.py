"""
Room Context Analyzer - Extracts style and context from room images
This is ADDITIVE - does not modify existing detection pipeline
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class RoomContextAnalyzer:
    """Analyzes room images for style, color, and spatial context"""
    
    def __init__(self):
        self.style_keywords = {
            'modern': ['clean lines', 'minimal', 'contemporary', 'sleek'],
            'traditional': ['ornate', 'classic', 'vintage', 'antique'],
            'industrial': ['exposed', 'metal', 'concrete', 'raw'],
            'scandinavian': ['white', 'wood', 'simple', 'nordic'],
            'bohemian': ['eclectic', 'colorful', 'layered', 'artistic']
        }
    
    def analyze_room_context(self, image: np.ndarray, detected_objects: List[Dict]) -> Dict:
        """
        Extract room context without modifying existing detection results
        
        Args:
            image: Original room image (BGR)
            detected_objects: Existing detected objects from ObjectDetector
            
        Returns:
            Context dictionary with style, colors, and spatial info
        """
        context = {
            'dominant_colors': self._extract_color_palette(image),
            'room_brightness': self._calculate_brightness(image),
            'object_density': len(detected_objects),
            'spatial_layout': self._analyze_spatial_layout(detected_objects, image.shape),
            'detected_styles': [],
            'room_type': self._infer_room_type(detected_objects)
        }
        
        return context
    
    def _extract_color_palette(self, image: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering"""
        try:
            # Resize for faster processing
            small = cv2.resize(image, (150, 150))
            pixels = small.reshape(-1, 3).astype(np.float32)
            
            # Simple K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Get color frequencies
            label_counts = Counter(labels.flatten())
            dominant_colors = []
            
            for idx, count in label_counts.most_common(n_colors):
                color = centers[idx].astype(int).tolist()
                dominant_colors.append(tuple(color))
            
            return dominant_colors
        except Exception as e:
            logger.warning(f"Failed to extract color palette: {e}")
            return [(128, 128, 128)]  # Default gray
    
    def _calculate_brightness(self, image: np.ndarray) -> str:
        """Calculate room brightness level"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 60:
                return "dark"
            elif mean_brightness < 120:
                return "medium"
            else:
                return "bright"
        except Exception as e:
            logger.warning(f"Failed to calculate brightness: {e}")
            return "medium"
    
    def _analyze_spatial_layout(self, objects: List[Dict], image_shape: Tuple) -> Dict:
        """Analyze spatial distribution of objects"""
        if not objects:
            return {'distribution': 'empty', 'clustering': 0.0}
        
        try:
            h, w = image_shape[:2]
            
            # Calculate object centers
            centers = []
            for obj in objects:
                if 'bbox' in obj:
                    x1, y1, x2, y2 = obj['bbox']
                    cx = (x1 + x2) / 2 / w  # Normalize
                    cy = (y1 + y2) / 2 / h
                    centers.append((cx, cy))
            
            # Simple clustering metric
            if len(centers) > 1:
                distances = []
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                     (centers[i][1] - centers[j][1])**2)
                        distances.append(dist)
                avg_distance = np.mean(distances)
                clustering = 1.0 - min(avg_distance, 1.0)
            else:
                clustering = 0.0
            
            return {
                'distribution': 'clustered' if clustering > 0.5 else 'spread',
                'clustering': float(clustering)
            }
        except Exception as e:
            logger.warning(f"Failed to analyze spatial layout: {e}")
            return {'distribution': 'unknown', 'clustering': 0.5}
    
    def _infer_room_type(self, objects: List[Dict]) -> str:
        """Infer room type from detected objects"""
        room_indicators = {
            'bedroom': ['bed', 'nightstand', 'wardrobe', 'dresser'],
            'living_room': ['sofa', 'couch', 'coffee table', 'tv', 'television'],
            'dining_room': ['dining table', 'chair', 'chandelier'],
            'kitchen': ['refrigerator', 'oven', 'microwave', 'counter'],
            'office': ['desk', 'office chair', 'monitor', 'bookshelf']
        }
        
        scores = {room: 0 for room in room_indicators}
        
        for obj in objects:
            obj_class = obj.get('class', '').lower()
            for room, indicators in room_indicators.items():
                if any(indicator in obj_class for indicator in indicators):
                    scores[room] += 1
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'general'