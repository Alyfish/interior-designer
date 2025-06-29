import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
import logging
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict

# SAM detector currently not implemented
SAM_AVAILABLE = False

logger = logging.getLogger(__name__)

class SegBackend(Enum):
    YOLOV8 = "yolov8"
    MASK2FORMER = "mask2former"
    COMBINED = "combined"

# Determine the project root directory dynamically
# This assumes new_object_detector.py is in interior_designer/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "yolov8x-seg.pt"
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_NAME)

class ObjectDetector:
    def __init__(self, model_path: str = MODEL_PATH, backend: SegBackend = SegBackend.MASK2FORMER):
        """
        Initialize object detector with specified backend.
        
        Args:
            model_path: Path to YOLO model (used only for YOLOV8 backend)
            backend: Detection backend to use (YOLOV8 or MASK2FORMER)
        """
        self.model_path = model_path
        self.backend = backend
        self.model = None
        self.mask2former_detector = None
        self.sam_detector = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "segmented_images").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        
        if self.backend == SegBackend.YOLOV8:
            self._load_model()
        elif self.backend == SegBackend.MASK2FORMER:
            self._load_mask2former()
        elif self.backend == SegBackend.COMBINED:
            # Load both models for combined detection
            self._load_model()
            self._load_mask2former()

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
    
    def _load_mask2former(self):
        """Load Mask2Former detector"""
        try:
            from mask2former_detector import Mask2FormerDetector
            self.mask2former_detector = Mask2FormerDetector()
            logger.info("✅ Mask2Former detector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to load Mask2Former detector: {e}")
            raise

    def _convert_contour_to_python_ints(self, contour):
        """Convert contour coordinates to Python integers for JSON compatibility."""
        return [[int(point[0]), int(point[1])] for point in contour]
    
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
                    
                    # Convert contour to Python integers
                    python_contour = self._convert_contour_to_python_ints(contour)
                    
                    walls_and_floors.append({
                        'id': f'wall_{i}',
                        'class': 'wall',
                        'confidence': confidence,
                        'contours': [python_contour],
                        'bbox': [min([p[0] for p in python_contour]), min([p[1] for p in python_contour]),
                                max([p[0] for p in python_contour]), max([p[1] for p in python_contour])],
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
                    
                    # Convert floor contour to Python integers
                    python_floor_contour = self._convert_contour_to_python_ints(floor_contour)
                    
                    walls_and_floors.append({
                        'id': 'floor_0',
                        'class': 'floor',
                        'confidence': 0.75,
                        'contours': [python_floor_contour],
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
                    
                    # Convert fallback floor contour to Python integers  
                    python_fallback_contour = self._convert_contour_to_python_ints(floor_contour)
                    
                    walls_and_floors.append({
                        'id': 'floor_fallback',
                        'class': 'floor',
                        'confidence': 0.6,
                        'contours': [python_fallback_contour],
                        'bbox': [0, floor_y, w, h],
                        'source': 'geometric_analysis'
                    })
                    
                logger.info(f"✅ Detected {len(walls_and_floors)} walls/floors using enhanced geometric analysis")
            else:
                logger.warning("No lines detected for wall/floor analysis")
            
        except Exception as e:
            logger.error(f"Error in enhanced wall/floor detection: {e}")
        
        return walls_and_floors

    def detect_objects(self, image_path: str, progress_callback=None) -> tuple[np.ndarray, list, str]:
        """
        Detect objects in an image using the configured backend.

        Args:
            image_path: Path to the input image.
            progress_callback: Optional callback for progress updates (Mask2Former only)

        Returns:
            A tuple containing:
                - original_image_cv: The original image as a NumPy array (BGR).
                - detected_objects: A list of dictionaries, each representing a detected object 
                                    with 'class', 'confidence', 'bbox', and 'contours'.
                - segmented_image_path: Path to the saved image with segmentations.
        """
        print(f"🔍 ObjectDetector: Starting detection with {self.backend.value} backend")
        print(f"📁 Processing image: {image_path}")
        
        if self.backend == SegBackend.MASK2FORMER:
            try:
                print("🔧 Using Mask2Former API detection...")
                # Try Mask2Former first with progress callback
                result = self.mask2former_detector.detect_objects(image_path, progress_callback=progress_callback)
                print("✅ Mask2Former detection completed successfully")
                return result
            except TimeoutError as e:
                print(f"⏱️ Mask2Former timed out, falling back to YOLOv8...")
                logger.warning(f"⏱️ Mask2Former timed out after {str(e).split()[-1]}, falling back to YOLOv8")
                if progress_callback:
                    progress_callback("⏱️ Mask2Former timed out, switching to YOLOv8...")
                # Fall back to YOLO
                if not self.model:
                    self._load_model()
                return self._detect_with_yolo(image_path)
            except Exception as e:
                print(f"❌ Mask2Former failed: {e}, falling back to YOLOv8...")
                logger.warning(f"❌ Mask2Former failed: {e}, falling back to YOLOv8")
                if progress_callback:
                    progress_callback("🔄 Mask2Former unavailable, using YOLOv8...")
                # Fall back to YOLO
                if not self.model:
                    self._load_model()
                return self._detect_with_yolo(image_path)
        elif self.backend == SegBackend.COMBINED:
            print("🔧 Using COMBINED detection (YOLOv8 + Mask2Former)...")
            # Use combined detection
            result = self._detect_with_combined(image_path, progress_callback)
            print("✅ Combined detection completed")
            return result
        else:
            print("🔧 Using YOLOv8 local detection...")
            # Use YOLO backend
            result = self._detect_with_yolo(image_path)
            print("✅ YOLOv8 detection completed")
            return result
    
    def _detect_with_yolo(self, image_path: str) -> tuple[np.ndarray, list, str]:
        """
        Detect objects using YOLOv8.
        """
        try:
            original_image_pil = Image.open(image_path).convert("RGB")
            original_image_cv = np.array(original_image_pil)[:, :, ::-1] # RGB to BGR for OpenCV
        except FileNotFoundError:
            print(f"❌ Error: Image file not found at {image_path}")
            raise
        except Exception as e:
            print(f"❌ Error opening image {image_path}: {e}")
            raise

        results = self.model(original_image_pil, verbose=False) # verbose=False to reduce console spam
        
        detected_objects = []
        annotated_image = original_image_cv.copy()

        if results and results[0].masks is not None:
            masks = results[0].masks.xy  # Segmentation masks as polygons
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
            confs = results[0].boxes.conf.cpu().numpy()    # Confidences
            class_ids = results[0].boxes.cls.cpu().numpy() # Class IDs
            class_names = results[0].names # Class names dictionary

            for i, mask_contours in enumerate(masks):
                class_id = int(class_ids[i])
                class_name = class_names.get(class_id, "unknown")
                confidence = float(confs[i])
                bbox_coords = [int(coord) for coord in boxes[i]] # x1, y1, x2, y2

                # Convert mask to binary image for better contour extraction
                # Create a binary mask from the polygon
                mask_img = np.zeros(original_image_cv.shape[:2], dtype=np.uint8)
                pts = np.array(mask_contours, dtype=np.int32)
                cv2.fillPoly(mask_img, [pts], 255)
                
                # Find contours with CHAIN_APPROX_NONE for pixel-level accuracy
                contours_full, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Convert contours to proper format with regular Python ints
                contours_list = []
                for contour in contours_full:
                    # Only simplify slightly to preserve detail while reducing points
                    epsilon = 0.001 * cv2.arcLength(contour, True)  # Very small epsilon for minimal simplification
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    contour_points = approx.reshape(-1, 2)
                    # Ensure each point is [int, int] not numpy types
                    python_contour = [[int(point[0]), int(point[1])] for point in contour_points]
                    contours_list.append(python_contour)
                
                # Also keep original contour for drawing
                contours_cv = contours_full
                
                detected_objects.append({
                    "id": f"obj_{i}", # Simple ID for now
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox_coords, # [x1, y1, x2, y2]
                    "contours": contours_list, # Store as list of [x,y] points with proper ints
                    "source": "yolov8"
                })

                # Draw segmentation mask
                cv2.drawContours(annotated_image, contours_cv, -1, (0, 255, 0), 2) # Green contour
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox_coords
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue bounding box
                
                # Put label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            print("No objects detected or no masks available.")

        # Add wall and floor detection
        walls_and_floors = self.detect_walls_and_floors(original_image_cv)
        detected_objects.extend(walls_and_floors)


        # Save the annotated image
        output_dir = self.output_dir / "segmented_images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000) # Python equivalent for milliseconds timestamp
        segmented_image_filename = f"segmented_{timestamp}.jpg"
        segmented_image_path = output_dir / segmented_image_filename
        
        try:
            cv2.imwrite(str(segmented_image_path), annotated_image)
            logger.info(f"✅ Segmented image saved to {segmented_image_path}")
            logger.info(f"✅ Detected {len(detected_objects)} total objects (including walls/floors)")
        except Exception as e:
            logger.error(f"❌ Error saving segmented image: {e}")
            # Fallback path if saving fails, though this shouldn't happen with proper permissions
            segmented_image_path = self.output_dir / "segmented_images" / "default_segmented.jpg"
            cv2.imwrite(str(segmented_image_path), annotated_image) # Try again with a default name

        return original_image_cv, detected_objects, str(segmented_image_path)
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)
        
        if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
            return 0.0
        
        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area
        
        return intersect_area / union_area if union_area > 0 else 0.0
    
    def _merge_detection_results(self, yolo_objects, mask2former_objects, iou_threshold=0.5):
        """
        Merge YOLO and Mask2Former detection results intelligently.
        
        Args:
            yolo_objects: List of objects detected by YOLO
            mask2former_objects: List of objects detected by Mask2Former
            iou_threshold: IoU threshold for considering objects as duplicates
            
        Returns:
            Merged list of detected objects
        """
        merged_objects = []
        used_mask2former = set()
        
        # First, process YOLO objects
        for yolo_obj in yolo_objects:
            best_match = None
            best_iou = 0.0
            
            # Find best matching Mask2Former object
            for i, m2f_obj in enumerate(mask2former_objects):
                if i in used_mask2former:
                    continue
                    
                # Skip walls/floors from YOLO if Mask2Former detected them
                if yolo_obj['class'] in ['wall', 'floor'] and m2f_obj['class'] in ['wall', 'floor']:
                    continue
                    
                iou = self._calculate_iou(yolo_obj['bbox'], m2f_obj['bbox'])
                
                # Check if classes are similar or compatible
                class_compatible = (
                    yolo_obj['class'] == m2f_obj['class'] or
                    (yolo_obj['class'] in ['couch', 'sofa'] and m2f_obj['class'] in ['couch', 'sofa']) or
                    (yolo_obj['class'] in ['dining table', 'table'] and m2f_obj['class'] in ['table', 'desk'])
                )
                
                if iou > iou_threshold and class_compatible and iou > best_iou:
                    best_match = i
                    best_iou = iou
            
            if best_match is not None:
                # Use Mask2Former version (better segmentation quality)
                used_mask2former.add(best_match)
                m2f_obj = mask2former_objects[best_match].copy()
                # Keep YOLO's confidence if higher
                if yolo_obj['confidence'] > m2f_obj.get('confidence', 0):
                    m2f_obj['confidence'] = yolo_obj['confidence']
                m2f_obj['merged_from'] = 'both'
                m2f_obj['original_source'] = f"{yolo_obj['source']},{m2f_obj['source']}"
                merged_objects.append(m2f_obj)
            else:
                # Keep YOLO object
                yolo_obj = yolo_obj.copy()
                yolo_obj['merged_from'] = 'yolo_only'
                merged_objects.append(yolo_obj)
        
        # Add remaining Mask2Former objects (things YOLO missed)
        for i, m2f_obj in enumerate(mask2former_objects):
            if i not in used_mask2former:
                m2f_obj = m2f_obj.copy()
                m2f_obj['merged_from'] = 'mask2former_only'
                merged_objects.append(m2f_obj)
        
        # Update IDs to be unique
        for i, obj in enumerate(merged_objects):
            obj['id'] = f'combined_{i}'
        
        logger.info(f"Merged detection: {len(yolo_objects)} YOLO + {len(mask2former_objects)} Mask2Former = {len(merged_objects)} total")
        return merged_objects
    
    def _detect_with_combined(self, image_path: str, progress_callback=None) -> tuple[np.ndarray, list, str]:
        """
        Detect objects using combined YOLO + Mask2Former approach.
        
        First runs YOLO for fast initial detection, then Mask2Former to catch missed objects
        and improve segmentation quality.
        """
        try:
            # Load image
            original_image_pil = Image.open(image_path).convert("RGB")
            original_image_cv = np.array(original_image_pil)[:, :, ::-1]  # RGB to BGR
            
            # Step 1: Run YOLO detection
            if progress_callback:
                progress_callback("🤖 Step 1/2: Running YOLOv8 for initial detection...")
            
            # Run YOLO detection (without saving intermediate image)
            results = self.model(original_image_pil, verbose=False)
            yolo_objects = []
            
            if results and results[0].masks is not None:
                masks = results[0].masks.xy
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                class_names = results[0].names
                
                for i, mask_contours in enumerate(masks):
                    class_id = int(class_ids[i])
                    class_name = class_names.get(class_id, "unknown")
                    confidence = float(confs[i])
                    bbox_coords = [int(coord) for coord in boxes[i]]
                    
                    # Convert contours for hover functionality
                    contours_cv = [np.array(mask_contours, dtype=np.int32).reshape((-1, 1, 2))]
                    contours_list = []
                    for c in contours_cv:
                        contour_points = c.reshape(-1, 2)
                        python_contour = [[int(point[0]), int(point[1])] for point in contour_points]
                        contours_list.append(python_contour)
                    
                    yolo_objects.append({
                        "id": f"yolo_{i}",
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox_coords,
                        "contours": contours_list,
                        "source": "yolov8"
                    })
            
            # Add YOLO's wall/floor detection
            walls_and_floors = self.detect_walls_and_floors(original_image_cv)
            yolo_objects.extend(walls_and_floors)
            
            if progress_callback:
                progress_callback(f"✅ YOLOv8 detected {len(yolo_objects)} objects")
            
            # Step 2: Run Mask2Former to catch missed objects
            mask2former_objects = []
            try:
                if progress_callback:
                    progress_callback("🎭 Step 2/2: Running Mask2Former to improve detection...")
                
                # Use reasonable timeout for combined mode
                _, m2f_objects, _ = self.mask2former_detector.detect_objects(
                    image_path, 
                    timeout=180.0,  # Increased timeout for better reliability
                    progress_callback=lambda msg: progress_callback(f"   {msg}") if progress_callback else None
                )
                mask2former_objects = m2f_objects
                
                if progress_callback:
                    progress_callback(f"✅ Mask2Former detected {len(mask2former_objects)} objects")
                    
            except TimeoutError as e:
                logger.warning(f"⏱️ Mask2Former timed out in combined mode: {e}")
                if progress_callback:
                    progress_callback(f"⏱️ Mask2Former timed out, using YOLO results only")
            except Exception as e:
                logger.warning(f"❌ Mask2Former failed in combined mode: {e}")
                if progress_callback:
                    progress_callback(f"⚠️ Mask2Former unavailable, using YOLO results only")
            
            # Step 3: Merge results intelligently
            if progress_callback:
                progress_callback("🔀 Merging detection results...")
            
            detected_objects = self._merge_detection_results(yolo_objects, mask2former_objects)
            
            # Create final annotated image
            annotated_image = original_image_cv.copy()
            
            for obj in detected_objects:
                # Draw with different colors based on source
                if obj.get('merged_from') == 'both':
                    color = (0, 255, 255)  # Yellow for merged
                elif obj.get('merged_from') == 'mask2former_only':
                    color = (255, 0, 255)  # Magenta for Mask2Former only
                else:
                    color = (0, 255, 0)  # Green for YOLO only
                
                # Draw contours
                if obj.get('contours'):
                    for contour in obj['contours']:
                        pts = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.drawContours(annotated_image, [pts], -1, color, 2)
                
                # Draw bounding box
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{obj['class']}: {obj.get('confidence', 1.0):.2f}"
                if obj.get('merged_from'):
                    label += f" ({obj['merged_from'].replace('_', ' ')})"
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save annotated image
            output_dir = self.output_dir / "segmented_images"
            timestamp = int(time.time() * 1000)
            segmented_image_filename = f"combined_{timestamp}.jpg"
            segmented_image_path = output_dir / segmented_image_filename
            
            cv2.imwrite(str(segmented_image_path), annotated_image)
            
            if progress_callback:
                progress_callback(f"✅ Combined detection complete: {len(detected_objects)} total objects")
            
            logger.info(f"✅ Combined detection saved to {segmented_image_path}")
            logger.info(f"✅ Detection breakdown: YOLO: {len(yolo_objects)}, Mask2Former: {len(mask2former_objects)}, Merged: {len(detected_objects)}")
            
            return original_image_cv, detected_objects, str(segmented_image_path)
            
        except Exception as e:
            logger.error(f"Error in combined detection: {e}")
            if progress_callback:
                progress_callback(f"❌ Combined detection failed: {e}")
            # Fall back to YOLO only
            return self._detect_with_yolo(image_path)

    def _merge_detections(self, yolo_objects: List[Dict], sam_objects: List[Dict]) -> List[Dict]:
        """
        Intelligently merge YOLO and SAM detections, avoiding duplicates.
        
        Args:
            yolo_objects: List of objects detected by YOLO
            sam_objects: List of objects detected by SAM/Mask2Former
            
        Returns:
            Merged list of detected objects
        """
        merged = []
        used_sam_indices = set()
        
        # Higher IoU threshold for considering objects as duplicates
        duplicate_iou_threshold = 0.3  # Lowered to catch more overlaps
        
        # First, add all YOLO objects - they have priority
        for yolo_obj in yolo_objects:
            merged.append(yolo_obj)
        
        # Then add SAM objects that don't overlap with YOLO objects
        for i, sam_obj in enumerate(sam_objects):
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
                    if merged_obj.get('source') in ['sam', 'mask2former']:
                        iou = self._calculate_iou(merged_obj['bbox'], sam_obj['bbox'])
                        if iou > duplicate_iou_threshold:
                            # Keep the one with higher confidence
                            if merged_obj.get('confidence', 0) >= sam_obj.get('confidence', 0):
                                overlaps_with_merged_sam = True
                                break
                
                if not overlaps_with_merged_sam:
                    merged.append(sam_obj)
        
        logger.info(f"Detection merge: {len(yolo_objects)} YOLO + {len(sam_objects)} SAM = {len(merged)} total")
        
        # Update IDs to be unique
        for i, obj in enumerate(merged):
            if not obj['id'].startswith('wall_') and not obj['id'].startswith('floor'):
                obj['id'] = f'merged_{i}'
        
        return merged

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