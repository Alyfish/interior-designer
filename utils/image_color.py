"""
Utilities for extracting color information from images
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict
from sklearn.cluster import KMeans
from collections import Counter

# Color name mapping - maps hue ranges to common color names
COLOR_NAMES = {
    (0, 10): "red",
    (10, 20): "orange",
    (20, 33): "yellow",
    (33, 78): "green",
    (78, 100): "teal",
    (100, 131): "blue",
    (131, 170): "purple",
    (170, 180): "pink",
}

# Material colors for furniture
MATERIAL_COLORS = {
    "walnut": "#5d4037",
    "oak": "#a1887f",
    "cherry": "#8d6e63",
    "mahogany": "#4e342e",
    "maple": "#d7ccc8",
    "teak": "#795548",
    "beech": "#bcaaa4",
    "pine": "#d7ccc8",
    "ash": "#e0e0e0",
    "cedar": "#6d4c41",
    "birch": "#efebe9",
    "ebony": "#212121",
    "black": "#212121",
    "white": "#ffffff",
    "gray": "#9e9e9e",
    "brown": "#795548",
    "beige": "#d7ccc8",
    "cream": "#fff8e1",
    "tan": "#d7ccc8", 
    "navy": "#263238",
    "green": "#2e7d32",
    "blue": "#1565c0",
    "red": "#b71c1c",
    "orange": "#e65100",
    "yellow": "#fbc02d",
    "purple": "#4a148c",
    "pink": "#d81b60",
}

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color code"""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color code to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_color_name(hsv: Tuple[float, float, float]) -> str:
    """
    Get a descriptive color name from HSV values
    
    Args:
        hsv: (hue, saturation, value) tuple with h in [0,180], s,v in [0,255]
        
    Returns:
        str: Color name (e.g., "navy-blue", "walnut-brown")
    """
    h, s, v = hsv
    
    # For very low saturation, it's a shade of gray
    if s < 30:
        if v < 50:
            return "charcoal-gray"
        elif v < 120:
            return "gray"
        elif v < 200:
            return "light-gray"
        else:
            return "white"
            
    # For very low value, it's close to black
    if v < 40:
        return "black"
        
    # Find base color name from hue
    color_name = "unknown"
    for (hue_min, hue_max), name in COLOR_NAMES.items():
        if hue_min <= h < hue_max:
            color_name = name
            break
            
    # Add modifiers based on saturation and value
    if s < 90:
        color_name = f"muted-{color_name}"
    
    if v < 100:
        color_name = f"dark-{color_name}"
    elif v > 200:
        color_name = f"light-{color_name}"
        
    # Map to furniture material colors if close
    if color_name in ["dark-orange", "muted-orange", "dark-red", "muted-red"]:
        if 10 <= h <= 30:
            return "walnut-brown"
    elif "yellow" in color_name and s < 150:
        return "oak"
    elif color_name == "dark-red":
        return "mahogany"
        
    return color_name

def extract_dominant_colors(image_path: str, k: int = 3) -> List[Tuple[str, str]]:
    """
    Extract dominant colors from an image using k-means clustering
    
    Args:
        image_path: Path to the image file
        k: Number of dominant colors to extract
        
    Returns:
        List of tuples (color_name, hex_code)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return [("unknown", "#808080")]
    
    # Convert to RGB for better color processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape to list of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    
    # Get the dominant colors
    colors = kmeans.cluster_centers_.astype(int)
    
    # Get cluster sizes
    labels = kmeans.labels_
    counter = Counter(labels)
    
    # Sort colors by cluster size
    percentages = [counter[i] / len(labels) for i in range(k)]
    sorted_colors = [(colors[i], percentages[i]) for i in range(k)]
    sorted_colors.sort(key=lambda x: x[1], reverse=True)
    
    # Convert colors to names and hex codes
    result = []
    for color, percentage in sorted_colors:
        # Convert RGB to HSV
        hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
        
        # Get color name
        color_name = get_color_name(hsv)
        hex_code = rgb_to_hex(tuple(color))
        
        result.append((color_name, hex_code))
    
    return result

def get_material_color_keywords(image_path: str) -> Dict[str, str]:
    """
    Extract material and color keywords from an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dict with keys 'material', 'color', and 'hex'
    """
    # Extract dominant colors
    colors = extract_dominant_colors(image_path, k=3)
    
    # Use the most dominant color
    if colors:
        primary_color_name, primary_color_hex = colors[0]
        
        # Extract material from color name if possible
        material = None
        if "-" in primary_color_name:
            parts = primary_color_name.split("-")
            if parts[-1] in ["brown", "gray"]:
                if "walnut" in primary_color_name:
                    material = "walnut"
                elif "oak" in primary_color_name:
                    material = "oak"
                else:
                    material = parts[-1]  # Use brown, gray, etc.
        
        # If we couldn't extract a material, use the color category
        if not material:
            material = primary_color_name.split("-")[-1]  # Use base color as material
            
        # Get a clean color name
        color = primary_color_name
        
        return {
            "material": material,
            "color": color,
            "hex": primary_color_hex,
            "all_colors": colors
        }
    
    return {
        "material": "unknown",
        "color": "unknown",
        "hex": "#808080",
        "all_colors": []
    } 