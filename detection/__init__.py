"""
Detection module for identifying furniture and architectural elements in images.
"""

from .yolo_detector import process_with_yolo, create_yolo_objects, create_semantic_map, detect_walls_and_floors 