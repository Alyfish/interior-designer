import cv2
import numpy as np
import os
import time
import logging
import base64
import requests
from PIL import Image
from io import BytesIO
import replicate
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class Mask2FormerDetector:
    def __init__(self, api_token: str = None):
        """
        Initialize Mask2Former detector using Replicate API.
        
        Args:
            api_token: Replicate API token. If not provided, will look for REPLICATE_API_TOKEN env var.
        """
        # Try to get from config module first, then environment
        self.api_token = api_token
        if not self.api_token:
            try:
                from config import REPLICATE_API_TOKEN
                self.api_token = REPLICATE_API_TOKEN
            except ImportError:
                pass
        
        if not self.api_token:
            self.api_token = os.getenv("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found in environment or provided")
        
        # Set the API token for replicate
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        
        self.model = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "segmented_images").mkdir(exist_ok=True)
        
        # ADE20K class names mapping (partial list of common indoor objects)
        self.ade20k_classes = {
            0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "tree",
            5: "ceiling", 6: "road", 7: "bed", 8: "windowpane", 9: "grass",
            10: "cabinet", 11: "sidewalk", 12: "person", 13: "earth", 14: "door",
            15: "table", 16: "mountain", 17: "plant", 18: "curtain", 19: "chair",
            20: "car", 21: "water", 22: "painting", 23: "sofa", 24: "shelf",
            25: "house", 26: "sea", 27: "mirror", 28: "rug", 29: "field",
            30: "armchair", 31: "seat", 32: "fence", 33: "desk", 34: "rock",
            35: "wardrobe", 36: "lamp", 37: "bathtub", 38: "railing", 39: "cushion",
            40: "base", 41: "box", 42: "column", 43: "signboard", 44: "chest of drawers",
            45: "counter", 46: "sand", 47: "sink", 48: "skyscraper", 49: "fireplace",
            50: "refrigerator", 51: "grandstand", 52: "path", 53: "stairs", 54: "runway",
            55: "case", 56: "pool table", 57: "pillow", 58: "screen door", 59: "stairway",
            60: "river", 61: "bridge", 62: "bookcase", 63: "blind", 64: "coffee table",
            65: "toilet", 66: "flower", 67: "book", 68: "hill", 69: "bench",
            70: "countertop", 71: "stove", 72: "palm", 73: "kitchen island", 74: "computer",
            75: "swivel chair", 76: "boat", 77: "bar", 78: "arcade machine", 79: "hovel",
            80: "bus", 81: "towel", 82: "light", 83: "truck", 84: "tower",
            85: "chandelier", 86: "awning", 87: "streetlight", 88: "booth", 89: "television",
            90: "airplane", 91: "dirt track", 92: "apparel", 93: "pole", 94: "land",
            95: "bannister", 96: "escalator", 97: "ottoman", 98: "bottle", 99: "buffet",
            100: "poster", 101: "stage", 102: "van", 103: "ship", 104: "fountain",
            105: "conveyer belt", 106: "canopy", 107: "washer", 108: "plaything", 109: "swimming pool",
            110: "stool", 111: "barrel", 112: "basket", 113: "waterfall", 114: "tent",
            115: "bag", 116: "minibike", 117: "cradle", 118: "oven", 119: "ball",
            120: "food", 121: "step", 122: "tank", 123: "trade name", 124: "microwave",
            125: "pot", 126: "animal", 127: "bicycle", 128: "lake", 129: "dishwasher",
            130: "screen", 131: "blanket", 132: "sculpture", 133: "hood", 134: "sconce",
            135: "vase", 136: "traffic light", 137: "tray", 138: "ashcan", 139: "fan",
            140: "pier", 141: "crt screen", 142: "plate", 143: "monitor", 144: "bulletin board",
            145: "shower", 146: "radiator", 147: "glass", 148: "clock", 149: "flag"
        }
        
    def _image_to_base64_data_uri(self, pil_image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI."""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def _parse_segmentation_output(self, segment_image_url: str, objects_list: list) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """
        Parse the indexed color segmentation image from Mask2Former with object labels.
        
        Args:
            segment_image_url: URL to the segmentation mask image
            objects_list: List of objects with color and label information
            
        Returns:
            masks: Dict mapping class_name to binary mask
            color_to_class: Dict mapping RGB tuple to class name
        """
        # Download the segmentation image
        response = requests.get(segment_image_url)
        response.raise_for_status()
        
        # Open as PIL Image and convert to numpy array
        seg_img = Image.open(BytesIO(response.content)).convert("RGB")
        seg_array = np.array(seg_img)
        
        masks = {}
        color_to_class = {}
        
        # Process each object with its corresponding color
        for obj in objects_list:
            color = obj["color"]  # [R, G, B] values
            label = obj["label"]
            
            # Create binary mask for this specific color
            mask = np.all(seg_array == color, axis=2).astype(np.uint8) * 255
            
            # Skip if mask is too small (< 256 pixels)
            if np.sum(mask > 0) < 256:
                continue
            
            color_key = tuple(color)
            color_to_class[color_key] = label
            
            # Use label as key, combine masks if same label appears multiple times
            if label not in masks:
                masks[label] = mask
            else:
                # Combine masks for same class
                masks[label] = np.maximum(masks[label], mask)
        
        return masks, color_to_class
    
    def _run_prediction_with_polling(self, image_uri: str, timeout: float, progress_callback: Optional[Callable[[str], None]] = None):
        """
        Run Mask2Former prediction with polling for completion.
        """
        try:
            if progress_callback:
                progress_callback("🚀 Starting Mask2Former prediction...")
            
            # Create client with explicit token
            client = replicate.Client(api_token=self.api_token)
            
            # Get model and version
            model = client.models.get("hassamdevsy/mask2former")
            versions = model.versions.list()
            
            if not versions:
                raise ValueError("No versions found for hassamdevsy/mask2former model")
            
            version = versions[0]  # Use latest version
            logger.info(f"Using model version: {version.id}")
            
            if progress_callback:
                progress_callback(f"📡 Using model version: {version.id[:12]}...")
            
            # Create prediction
            prediction = client.predictions.create(
                version=version.id,
                input={"image": image_uri}
            )
            
            logger.info(f"Prediction created with ID: {prediction.id}")
            if progress_callback:
                progress_callback(f"⏳ Prediction started (ID: {prediction.id[:12]}...)")
            
            # Poll for completion
            start_time = time.time()
            poll_interval = 2.0  # Start with 2 second intervals
            max_poll_interval = 10.0
            
            while prediction.status not in ["succeeded", "failed", "canceled"]:
                elapsed = time.time() - start_time
                
                if elapsed > timeout:
                    # Try to cancel the prediction
                    try:
                        client.predictions.cancel(prediction.id)
                    except:
                        pass
                    raise TimeoutError(f"Prediction timed out after {elapsed:.1f}s")
                
                # Update progress
                if progress_callback:
                    status_msg = f"🔄 Processing... ({elapsed:.1f}s) Status: {prediction.status}"
                    progress_callback(status_msg)
                
                time.sleep(poll_interval)
                
                # Refresh prediction status
                prediction = client.predictions.get(prediction.id)
                
                # Increase poll interval gradually
                poll_interval = min(poll_interval * 1.2, max_poll_interval)
            
            # Check final status
            if prediction.status == "failed":
                error_msg = prediction.error or "Unknown error occurred"
                raise RuntimeError(f"Prediction failed: {error_msg}")
            elif prediction.status == "canceled":
                raise RuntimeError("Prediction was canceled")
            
            if progress_callback:
                progress_callback("✅ Prediction completed successfully!")
            
            return prediction.output
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            if progress_callback:
                progress_callback(f"❌ Prediction failed: {str(e)}")
            raise
    
    def detect_objects(self, image_path: str, timeout: float = 180.0, progress_callback: Optional[Callable[[str], None]] = None) -> tuple[np.ndarray, list, str]:
        """
        Detect objects using Mask2Former via Replicate API.
        
        Args:
            image_path: Path to input image
            timeout: Maximum time to wait for API response (default: 120s)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (original_image, detected_objects, segmented_image_path)
        """
        print("🌐 Mask2Former: Starting API detection...")
        start_time = time.time()
        
        try:
            print("📁 Loading and preparing image...")
            # Load image
            original_image_pil = Image.open(image_path).convert("RGB")
            original_image_cv = np.array(original_image_pil)[:, :, ::-1]  # RGB to BGR
            
            # Convert to base64 data URI
            image_uri = self._image_to_base64_data_uri(original_image_pil)
            print("✅ Image converted to base64 format")
            
            if progress_callback:
                progress_callback("📸 Image prepared, calling Mask2Former API...")
            
            print("🌐 Calling Mask2Former API via Replicate...")
            logger.info("Calling Mask2Former API with polling...")
            
            # Use the new polling method with better error handling
            output = self._run_prediction_with_polling(image_uri, timeout, progress_callback)
            
            print("📦 Processing API response...")
            # The output should contain the segmentation result
            if not output or "segment" not in output or "objects" not in output:
                raise ValueError("Invalid output received from Mask2Former API")
            
            # Extract segment URL and objects list from API response
            segment_url = output["segment"]
            objects_list = output["objects"]
            
            print(f"🎯 API returned {len(objects_list)} object classes")
            logger.info(f"Received {len(objects_list)} object classes from Mask2Former")
            
            if progress_callback:
                progress_callback(f"🎯 Processing {len(objects_list)} detected objects...")
            
            print("🔧 Parsing segmentation masks...")
            # Parse the segmentation output
            masks_dict, color_mapping = self._parse_segmentation_output(segment_url, objects_list)
            
            detected_objects = []
            annotated_image = original_image_cv.copy()
            
            # Convert masks to contours for hover functionality
            for i, (class_name, mask) in enumerate(masks_dict.items()):
                # Find contours with CHAIN_APPROX_NONE for pixel-level accuracy
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if not contours:
                    continue
                
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Minimal simplification to preserve detail while reducing points
                epsilon = 0.001 * cv2.arcLength(largest_contour, True)  # Much smaller epsilon
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox = [x, y, x + w, y + h]
                
                # Convert contour to the format expected by hover functionality
                contour_points = approx.reshape(-1, 2).tolist()
                
                detected_objects.append({
                    "id": f"mask2former_{i}",
                    "class": class_name,
                    "confidence": 1.0,  # Mask2Former doesn't provide confidence
                    "bbox": bbox,
                    "contours": [contour_points],
                    "source": "mask2former"
                })
                
                # Draw on annotated image
                cv2.drawContours(annotated_image, [approx], -1, (0, 255, 0), 2)
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                label = f"{class_name}"
                cv2.putText(annotated_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Save annotated image
            timestamp = int(time.time() * 1000)
            segmented_image_filename = f"mask2former_segmented_{timestamp}.jpg"
            segmented_image_path = self.output_dir / "segmented_images" / segmented_image_filename
            
            cv2.imwrite(str(segmented_image_path), annotated_image)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Mask2Former detected {len(detected_objects)} objects in {total_time:.1f}s")
            logger.info(f"✅ Segmented image saved to {segmented_image_path}")
            
            if progress_callback:
                progress_callback(f"🎉 Complete! Detected {len(detected_objects)} objects in {total_time:.1f}s")
            
            return original_image_cv, detected_objects, str(segmented_image_path)
            
        except TimeoutError as e:
            elapsed = time.time() - start_time
            logger.warning(f"Mask2Former timeout after {elapsed:.1f}s: {e}")
            if progress_callback:
                progress_callback(f"⏰ Timeout after {elapsed:.1f}s - will fallback to YOLOv8")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Mask2Former error after {elapsed:.1f}s: {e}")
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)} - will fallback to YOLOv8")
            raise

if __name__ == "__main__":
    # Test the detector
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Create a test image
        test_image = "test_mask2former.jpg"
        if not os.path.exists(test_image):
            dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "Test Image", (250, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(test_image, dummy_img)
    
    try:
        detector = Mask2FormerDetector()
        image, objects, seg_path = detector.detect_objects(test_image)
        
        print(f"\nDetected {len(objects)} objects:")
        for obj in objects:
            print(f"  - Class: {obj['class']}, Confidence: {obj['confidence']:.2f}")
        print(f"Segmented image saved at: {seg_path}")
        
    except Exception as e:
        print(f"Error: {e}")