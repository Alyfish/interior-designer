import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
import logging
from pathlib import Path
from typing import List, Dict
from sam_detector import SAMDetector

logger = logging.getLogger(__name__)

# Determine the project root directory dynamically
# This assumes new_object_detector.py is in interior_designer/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "yolov8x-seg.pt"
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_NAME)

class ObjectDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize YOLO object detector with enhanced wall/floor detection
        """
        self.model_path = model_path
        self.model = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize SAM detector and enable it by default
        self.sam_detector = SAMDetector()
        self.use_sam = True  # Enable SAM detection by default
        
        # Create subdirectories
        (self.output_dir / "segmented_images").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        
        self._load_model()

    def set_sam_enabled(self, enabled: bool):
        """Enable or disable SAM detection"""
        self.use_sam = enabled
        logger.info(f"SAM detection {'enabled' if enabled else 'disabled'}")

    def _load_model(self):
        """Load YOLO model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"✅ YOLOv8 segmentation model loaded from {self.model_path}")
            else:
                # Download model if not exists
                logger.info(f"Model not found at {self.model_path}, downloading...")
                self.model = YOLO("yolov8x-seg.pt")
                logger.info("✅ YOLOv8 segmentation model downloaded and loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise

    def detect_walls_and_floors(self, image):
        """
        Enhanced wall and floor detection using edge detection and geometric analysis
        """
        walls_and_floors = []
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Enhanced edge detection with multiple scales
            edges1 = cv2.Canny(blurred, 30, 100, apertureSize=3)
            edges2 = cv2.Canny(blurred, 50, 150, apertureSize=3)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Morphological operations to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Detect lines using Hough Transform with more permissive parameters
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=int(min(w, h) * 0.15), maxLineGap=20)
            
            if lines is not None:
                logger.info(f"Found {len(lines)} potential lines")
                
                # Analyze lines to identify walls and floors
                vertical_lines = []
                horizontal_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate angle and length
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length < min(w, h) * 0.1:  # Skip short lines
                        continue
                        
                    angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                    
                    # Classify as vertical (walls) or horizontal (floor/ceiling)
                    if abs(angle) > 75 or abs(angle-180) < 15 or abs(angle+180) < 15:  # Vertical lines
                        vertical_lines.append((line[0], length))
                    elif abs(angle) < 15 or abs(angle-180) < 15:  # Horizontal lines
                        horizontal_lines.append((line[0], length))
                
                logger.info(f"Found {len(vertical_lines)} vertical lines, {len(horizontal_lines)} horizontal lines")
                
                # Create wall objects from vertical lines (prioritize longer lines)
                vertical_lines.sort(key=lambda x: x[1], reverse=True)  # Sort by length
                for i, ((x1, y1, x2, y2), length) in enumerate(vertical_lines[:4]):  # Limit to 4 walls
                    # Create a rectangular contour for the wall
                    wall_width = max(10, int(w * 0.01))  # Adaptive wall width
                    
                    # Ensure the line spans a reasonable height
                    if abs(y2 - y1) < h * 0.2:
                        continue
                    
                    if abs(x2 - x1) < abs(y2 - y1):  # More vertical than horizontal
                        center_x = (x1 + x2) // 2
                        contour = [
                            [max(0, center_x - wall_width//2), min(y1, y2)],
                            [min(w, center_x + wall_width//2), min(y1, y2)],
                            [min(w, center_x + wall_width//2), max(y1, y2)],
                            [max(0, center_x - wall_width//2), max(y1, y2)]
                        ]
                    else:
                        center_y = (y1 + y2) // 2
                        contour = [
                            [min(x1, x2), max(0, center_y - wall_width//2)],
                            [max(x1, x2), max(0, center_y - wall_width//2)],
                            [max(x1, x2), min(h, center_y + wall_width//2)],
                            [min(x1, x2), min(h, center_y + wall_width//2)]
                        ]
                    
                    # Calculate confidence based on line strength and position
                    edge_strength = cv2.mean(edges[max(0, min(y1, y2)):min(h, max(y1, y2)), 
                                                  max(0, min(x1, x2)):min(w, max(x1, x2))])[0]
                    confidence = min(0.9, 0.5 + (edge_strength / 255.0) * 0.4)
                    
                    walls_and_floors.append({
                        'id': f'wall_{i}',
                        'class': 'wall',
                        'confidence': confidence,
                        'contours': [contour],
                        'bbox': [min([p[0] for p in contour]), min([p[1] for p in contour]),
                                max([p[0] for p in contour]), max([p[1] for p in contour])],
                        'source': 'geometric_analysis'
                    })
                
                # Create floor object from horizontal lines and image analysis
                horizontal_lines.sort(key=lambda x: x[1], reverse=True)  # Sort by length
                floor_candidates = []
                
                # Look for floor lines in the bottom half of the image
                for (x1, y1, x2, y2), length in horizontal_lines:
                    if max(y1, y2) > h * 0.5:  # In bottom half
                        floor_candidates.append((x1, y1, x2, y2, length))
                
                if floor_candidates:
                    # Use the longest horizontal line in the bottom area as floor reference
                    x1, y1, x2, y2, length = max(floor_candidates, key=lambda x: x[4])
                    floor_y = max(y1, y2)
                    
                    # Create floor contour (bottom portion of image)
                    floor_height = max(20, int(h * 0.15))  # At least 15% of image height
                    floor_contour = [
                        [0, max(0, floor_y - floor_height//2)],
                        [w, max(0, floor_y - floor_height//2)],
                        [w, h],
                        [0, h]
                    ]
                    
                    walls_and_floors.append({
                        'id': 'floor_0',
                        'class': 'floor',
                        'confidence': 0.75,
                        'contours': [floor_contour],
                        'bbox': [0, max(0, floor_y - floor_height//2), w, h],
                        'source': 'geometric_analysis'
                    })
                else:
                    # Fallback: assume bottom 20% is floor
                    floor_y = int(h * 0.8)
                    floor_contour = [
                        [0, floor_y],
                        [w, floor_y],
                        [w, h],
                        [0, h]
                    ]
                    
                    walls_and_floors.append({
                        'id': 'floor_fallback',
                        'class': 'floor',
                        'confidence': 0.6,
                        'contours': [floor_contour],
                        'bbox': [0, floor_y, w, h],
                        'source': 'geometric_analysis'
                    })
                    
                logger.info(f"✅ Detected {len(walls_and_floors)} walls/floors using enhanced geometric analysis")
            else:
                logger.warning("No lines detected for wall/floor analysis")
            
        except Exception as e:
            logger.error(f"Error in enhanced wall/floor detection: {e}")
        
        return walls_and_floors

    def detect_objects(self, image_path: str) -> tuple[np.ndarray, list, str]:
        """Enhanced object detection with coordinate validation"""
        try:
            # Get timeout from session state or use default
            sam_timeout_minutes = 5  # Default timeout
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'sam_timeout'):
                    sam_timeout_minutes = max(st.session_state.sam_timeout // 60, 3)  # At least 3 minutes
            except:
                pass
                
            # Load image
            original_image_pil = Image.open(image_path).convert("RGB")
            original_image_cv = np.array(original_image_pil)[:, :, ::-1]
            
            # Store original dimensions
            orig_height, orig_width = original_image_cv.shape[:2]
            logger.info(f"Original image dimensions: {orig_width}x{orig_height}")

            # Run YOLO detection
            results = self.model(original_image_pil, verbose=False)
            
            detected_objects = []
            if results and results[0].masks is not None:
                masks = results[0].masks.xy
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                class_names_map = results[0].names

                for i, mask_contours in enumerate(masks):
                    detected_objects.append({
                        "id": f"yolo_{i}",
                        "class": class_names_map.get(int(class_ids[i]), "unknown"),
                        "confidence": float(confs[i]),
                        "bbox": [int(c) for c in boxes[i]],
                        "contours": [mask_contours.tolist()],
                        'source': 'yolo'
                    })
            logger.info(f"✅ YOLOv8 detected {len(detected_objects)} objects")

            # Get wall and floor detections
            walls_and_floors = self.detect_walls_and_floors(original_image_cv)
            
            # Validate and fix coordinates for all objects
            all_objects = detected_objects + walls_and_floors
            
            for obj in all_objects:
                # Ensure bbox is within image bounds
                if 'bbox' in obj and len(obj['bbox']) == 4:
                    x1, y1, x2, y2 = obj['bbox']
                    obj['bbox'] = [
                        max(0, min(x1, orig_width)),
                        max(0, min(y1, orig_height)),
                        max(0, min(x2, orig_width)),
                        max(0, min(y2, orig_height))
                    ]
                
                # Ensure contours are within bounds
                if 'contours' in obj:
                    validated_contours = []
                    for contour in obj['contours']:
                        if isinstance(contour, list):
                            validated_contour = []
                            for point in contour:
                                if isinstance(point, (list, tuple)) and len(point) >= 2:
                                    validated_point = [
                                        max(0, min(int(point[0]), orig_width)),
                                        max(0, min(int(point[1]), orig_height))
                                    ]
                                    validated_contour.append(validated_point)
                            if validated_contour:
                                validated_contours.append(validated_contour)
                    obj['contours'] = validated_contours

            # Add SAM detection if enabled
            if self.sam_detector.enabled:
                try:
                    sam_objects = self.sam_detector.detect_objects(image_path)
                    merged_objects = self._merge_detections(all_objects, sam_objects)
                    all_objects = merged_objects
                except Exception as e:
                    logger.error(f"SAM detection failed: {e}")
            
            logger.info(f"✅ Total detected objects: {len(all_objects)}")

            # Draw segmentation on a copy of the original image
            segmented_image = original_image_cv.copy()
            logger.info(f"🎨 Drawing {len(all_objects)} objects on {orig_width}x{orig_height} canvas")
            self.draw_segmentation(segmented_image, all_objects)

            # Save the segmented image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            segmented_image_path = self.output_dir / "segmented_images" / f"segmented_{timestamp}.jpg"
            cv2.imwrite(str(segmented_image_path), segmented_image)
            logger.info(f"✅ Segmented image saved to {segmented_image_path}")

            return original_image_cv, all_objects, str(segmented_image_path)
            
        except Exception as e:
            logger.error(f"Error in detect_objects: {e}", exc_info=True)
            # Fallback to avoid crashing the app
            placeholder_img = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "Error processing image", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return placeholder_img, [], ""

    def _merge_detections(self, yolo_objects: List[Dict], sam_objects: List[Dict]) -> List[Dict]:
        """
        Intelligently merge YOLO and SAM detections, avoiding duplicates
        """
        merged = []
        used_sam_indices = set()
        used_yolo_indices = set()
        
        # Define furniture and interior design related classes we care about
        furniture_classes = {
            'chair', 'couch', 'sofa', 'table', 'desk', 'bed', 'cabinet', 
            'shelf', 'bookshelf', 'dresser', 'nightstand', 'ottoman', 
            'bench', 'stool', 'armchair', 'wardrobe', 'drawer',
            'lamp', 'rug', 'carpet', 'mirror', 'cushion', 'pillow',
            'coffee table', 'dining table', 'side table', 'end table',
            'entertainment center', 'tv stand', 'plant', 'vase', 'painting',
            'refrigerator', 'television', 'tv', 'blanket', 'curtain',
            'door', 'window', 'floor', 'wall', 'ceiling'
        }
        
        # Higher IoU threshold for considering objects as duplicates
        duplicate_iou_threshold = 0.3  # Lowered to catch more overlaps
        
        # First, add all YOLO objects - they have priority
        for yolo_obj in yolo_objects:
            merged.append(yolo_obj)
        
        # Then add SAM objects that don't overlap with YOLO objects
        for sam_obj in sam_objects:
            # Check if this SAM object overlaps significantly with any YOLO object
            overlaps_with_yolo = False
            
            for yolo_obj in yolo_objects:
                iou = self._calculate_iou(yolo_obj['bbox'], sam_obj['bbox'])
                if iou > duplicate_iou_threshold:
                    overlaps_with_yolo = True
                    logger.info(f"Skipping SAM {sam_obj['class']} - overlaps with YOLO {yolo_obj['class']} (IoU={iou:.2f})")
                    break
            
            # Also check if it overlaps with already added SAM objects
            if not overlaps_with_yolo:
                overlaps_with_merged_sam = False
                for merged_obj in merged:
                    if merged_obj['source'] == 'sam':
                        iou = self._calculate_iou(merged_obj['bbox'], sam_obj['bbox'])
                        if iou > duplicate_iou_threshold:
                            # Keep the one with higher confidence
                            if merged_obj.get('confidence', 0) >= sam_obj.get('confidence', 0):
                                overlaps_with_merged_sam = True
                                break
                
                if not overlaps_with_merged_sam:
                    merged.append(sam_obj)
        

        
        logger.info(f"Detection merge: {len(yolo_objects)} YOLO + {len(sam_objects)} SAM = {len(merged)} total")
        
        # Log class distribution
        class_counts = {}
        for obj in merged:
            class_name = obj['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        logger.info(f"Merged object classes: {class_counts}")
        
        return merged

    def _are_classes_compatible(self, yolo_class: str, sam_class: str) -> bool:
        # This is a placeholder for more advanced logic if needed
        return True

    def _is_more_specific(self, specific_class: str, general_class: str) -> bool:
        # This is a placeholder for more advanced logic if needed
        return len(specific_class) > len(general_class)

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def draw_segmentation(self, image, detected_objects):
        # This method needs to be implemented to draw segmentations on the image
        # It's called in the detect_objects method
        pass

if __name__ == '__main__':
    print("Running Object Detector Test...")
    # Create a dummy image for testing if no image is provided
    test_image_path = os.path.join(PROJECT_ROOT, "input", "test_image.jpg") 
    # Fallback: use a known image from the project if the above doesn't exist
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Trying 'test_img.jpg' in project root.")
        test_image_path = os.path.join(PROJECT_ROOT, "test_img.jpg")

    if not os.path.exists(test_image_path):
        print(f"❌ Test image {test_image_path} not found. Creating a dummy image.")
        os.makedirs(os.path.join(PROJECT_ROOT, "input"), exist_ok=True)
        dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Test Image", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(test_image_path, dummy_img)
        print(f"✅ Dummy test image created at {test_image_path}")

    detector = ObjectDetector()
    try:
        image, objects, seg_path = detector.detect_objects(test_image_path)
        print(f"\nDetected {len(objects)} objects:")
        for obj in objects:
            print(f"  - Class: {obj['class']}, Confidence: {obj['confidence']:.2f}, BBox: {obj['bbox']}")
        print(f"Segmented image saved at: {seg_path}")
        
        # Display the image (optional, requires GUI environment)
        # cv2.imshow("Original Image", image)
        # annotated_display = cv2.imread(seg_path)
        # cv2.imshow("Segmented Image", annotated_display)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ An error occurred during detection test: {e}")

    print("Object Detector Test Finished.") 