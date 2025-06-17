import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
import logging
from pathlib import Path

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
        
        # Create subdirectories
        (self.output_dir / "segmented_images").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        
        self._load_model()

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
        """
        Detect objects in an image using YOLOv8, draws segmentations and bounding boxes.

        Args:
            image_path: Path to the input image.

        Returns:
            A tuple containing:
                - original_image_cv: The original image as a NumPy array (BGR).
                - detected_objects: A list of dictionaries, each representing a detected object 
                                    with 'class', 'confidence', 'bbox', and 'contours'.
                - segmented_image_path: Path to the saved image with segmentations.
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

                # Convert mask_contours (list of np.array) to list of lists for JSON serializability if needed
                # and for consistency in drawing.
                # Ensure contours are in the correct format for OpenCV drawing (list of np arrays of points)
                contours_cv = [np.array(contour, dtype=np.int32).reshape((-1, 1, 2)) for contour in [mask_contours]]

                detected_objects.append({
                    "id": f"obj_{i}", # Simple ID for now
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox_coords, # [x1, y1, x2, y2]
                    "contours": [c.reshape(-1, 2).tolist() for c in contours_cv] # Store as list of [x,y] points
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