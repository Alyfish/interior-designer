import logging
import os
import requests
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import replicate
import json
from pycocotools import mask as mask_utils
from config import REPLICATE_API_TOKEN, IMGBB_API_KEY
import base64
from io import BytesIO
import time
import threading

logger = logging.getLogger(__name__)

# Try to import config, but fall back to environment variables
try:
    from config import REPLICATE_API_TOKEN
except ImportError:
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

class SAMDetector:
    # Class-level shared state to prevent multiple concurrent requests
    _global_prediction_id = None
    _global_prediction_lock = threading.Lock()
    _global_prediction_start_time = None
    
    def __init__(self):
        """Initialize Semantic Segment Anything detector"""
        # Enable SAM if we have a Replicate token and detection hasn't been disabled explicitly
        self.enabled = bool(REPLICATE_API_TOKEN) and os.getenv("ENABLE_SAM_DETECTION", "on").lower() in ["on", "true", "1"]

        if self.enabled:
            os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
            logger.info("✅ SAM detector initialized with Replicate API")
        else:
            logger.warning("⚠️ SAM detector disabled (missing token or explicitly turned off)")

        # Use the semantic-segment-anything model from the documentation
        self._model_slug = os.getenv("SAM_MODEL_SLUG", "cjwbw/semantic-segment-anything")
        # Allow overriding a specific version via env var
        self._fixed_version_id = os.getenv("SAM_MODEL_VERSION")
        self._cached_version_id = None  # Lazy-fetched latest version
        
        # Instance variables to track image scaling for coordinate correction
        self.resize_ratio = 1.0
        self.original_size = None
        self.resized_size = None

    @classmethod
    def _cancel_global_prediction(cls):
        """Cancel any existing global prediction."""
        with cls._global_prediction_lock:
            if cls._global_prediction_id:
                try:
                    logger.info(f"🛑 Canceling global prediction: {cls._global_prediction_id}")
                    replicate.predictions.cancel(cls._global_prediction_id)
                except Exception as e:
                    logger.warning(f"Failed to cancel global prediction {cls._global_prediction_id}: {e}")
                finally:
                    cls._global_prediction_id = None
                    cls._global_prediction_start_time = None
    
    @classmethod
    def _is_prediction_running(cls) -> bool:
        """Check if a prediction is running globally."""
        with cls._global_prediction_lock:
            return cls._global_prediction_id is not None

    def _should_keep_object(self, annotation: Dict[str, Any]) -> bool:
        """Determine if an object is worth keeping based on its properties."""
        area = annotation.get('area', 0)
        if area < 400:  # Filter out very small, noisy objects
            return False
        
        confidence = annotation.get('predicted_iou', 0)
        stability_score = annotation.get('stability_score', 0)
        if confidence < 0.4 and stability_score < 0.4:
            return False
            
        return True

    def _prepare_image_for_replicate(self, image_path: str) -> str:
        """Resize image for SAM and encode as a base64 data URL."""
        temp_path = None
        try:
            with Image.open(image_path) as img:
                # Handle images with alpha channels
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                self.original_size = img.size
                logger.info(f"Original image size: {self.original_size}")
                
                max_size = 1024
                if max(img.size) > max_size:
                    self.resize_ratio = max_size / max(img.size)
                    self.resized_size = (int(img.width * self.resize_ratio), int(img.height * self.resize_ratio))
                    img = img.resize(self.resized_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image for SAM to: {self.resized_size} (ratio: {self.resize_ratio:.3f})")
                else:
                    self.resize_ratio = 1.0
                    self.resized_size = self.original_size

                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=90)
                encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded_string}"

        except Exception as e:
            logger.error(f"Error preparing image for Replicate: {e}", exc_info=True)
            raise

    def _rle_to_contours(self, rle_mask: Dict, display_size: tuple) -> List[List[List[int]]]:
        """Convert COCO RLE to OpenCV contours, handling size mismatches."""
        try:
            binary_mask = mask_utils.decode(rle_mask)
            mask_h, mask_w = binary_mask.shape
            display_w, display_h = display_size
            
            if (mask_w, mask_h) != (display_w, display_h):
                logger.info(f"Resizing mask from {mask_w}x{mask_h} to {display_w}x{display_h} for display accuracy.")
                binary_mask = cv2.resize(binary_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return [[[int(point[0][0]), int(point[0][1])] for point in c] for c in contours if len(c) >= 3]
            
        except Exception as e:
            logger.error(f"Error converting RLE to contours: {e}", exc_info=True)
            return []

    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """Run SAM detection using the simpler replicate.run() API."""
        if not self.enabled:
            return []

        try:
            image_data_url = self._prepare_image_for_replicate(image_path)
            
            logger.info("Running SAM detection using replicate.run()...")
            start_time = time.time()
            
            # Use the exact format from the documentation
            output = replicate.run(
                "cjwbw/semantic-segment-anything:b2691db53f2d96add0051a4a98e7a3861bd21bf5972031119d344d956d2f8256",
                input={
                    "image": image_data_url
                }
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ SAM prediction completed in {elapsed:.1f} seconds!")
            
            return self._process_sam_output(output, image_path)
            
        except Exception as e:
            logger.error(f"SAM detection failed: {e}", exc_info=True)
            return []

    def _poll_for_prediction_completion(self, original_image_path: str) -> List[Dict[str, Any]]:
        """Poll Replicate API for prediction results with timeouts."""
        try:
            import streamlit as st
            max_wait_time = st.session_state.get('sam_timeout', 300)
        except (ImportError, AttributeError):
            max_wait_time = 300
            
        max_polls = max_wait_time // 2
        
        for poll_count in range(max_polls):
            elapsed = time.time() - self._global_prediction_start_time
            if elapsed > max_wait_time:
                logger.error(f"⏰ SAM prediction timed out after {elapsed:.1f} seconds")
                break

            try:
                prediction = replicate.predictions.get(self._global_prediction_id)
                if poll_count % 10 == 0:
                     logger.info(f"Poll {poll_count}/{max_polls}: Status = {prediction.status} (elapsed: {elapsed:.1f}s)")
                
                if prediction.status == "succeeded":
                    logger.info(f"✅ SAM prediction completed in {elapsed:.1f} seconds!")
                    return self._process_sam_output(prediction.output, original_image_path)
                elif prediction.status in ["failed", "canceled"]:
                    logger.error(f"❌ SAM prediction {prediction.status}: {prediction.error}")
                    break
            except Exception as e:
                logger.error(f"Error checking prediction status: {e}")

            time.sleep(2)

        self._cancel_global_prediction()
        return []

    def _process_sam_output(self, output: Any, original_image_path: str) -> List[Dict[str, Any]]:
        """Process the JSON output from SAM, scaling coordinates correctly."""
        try:
            # Log the output structure
            logger.info(f"SAM API returned type: {type(output)}")
            
            if isinstance(output, dict):
                logger.info(f"SAM API returned keys: {list(output.keys())}")
                
                # Get the JSON URL from the output
                json_url = output.get('json_out')
                if not json_url:
                    logger.warning("No json_out URL found in SAM output")
                    return []
                
                logger.info(f"Found json_out URL: {json_url}")
                
                # Download JSON from URL
                response = requests.get(json_url, timeout=30)
                response.raise_for_status()
                json_data = response.json()
                logger.info(f"Downloaded JSON data from URL, got {len(json_data)} items")
            elif isinstance(output, list):
                # Sometimes the output might be direct JSON data
                json_data = output
                logger.info(f"Got direct JSON data with {len(json_data)} items")
            else:
                logger.warning(f"Unexpected SAM output type: {type(output)}")
                return []

            logger.info(f"Processing {len(json_data)} annotations from SAM")
            detected_objects = []
            
            for idx, anno in enumerate(json_data):
                if not self._should_keep_object(anno):
                    continue
                
                # Get the best class name
                class_name = self._get_best_class_name(anno)
                
                # Skip non-furniture items
                if not self._is_furniture_or_decor(class_name):
                    continue
                
                # Get bbox and scale it if needed
                x, y, w, h = anno.get('bbox', [0,0,0,0])
                if self.resize_ratio != 1.0:
                    scale = 1.0 / self.resize_ratio
                    x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

                # Convert segmentation to contours
                contours = self._rle_to_contours(anno.get('segmentation'), self.original_size)
                if not contours:
                    continue

                iou_score = min(1.0, float(anno.get('predicted_iou', 0.0)))  # Cap at 1.0
                area = int(anno.get('area', 0))
                
                logger.info(f"Added furniture/decor: {class_name} (confidence={iou_score:.2f}, area={area})")
                
                detected_objects.append({
                    "class": class_name.title(),
                    'id': f'sam_{idx}',
                    'confidence': iou_score,
                    'bbox': [x, y, x + w, y + h],  # Convert to [x1, y1, x2, y2] format
                    'contours': contours,
                    'area': area,
                    'source': 'sam'
                })
            
            logger.info(f"Successfully processed {len(detected_objects)} objects from SAM")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error processing SAM output: {e}", exc_info=True)
            return []

    def _get_best_class_name(self, annotation: Dict[str, Any]) -> str:
        """Get the most descriptive class name from annotation data."""
        proposals = annotation.get('class_proposals', [])
        if proposals:
            proposals.sort(key=len, reverse=True)
            for prop in proposals:
                if prop.lower() not in ['other', 'thing', 'stuff']:
                    return self._clean_class_name(prop)
        class_name = annotation.get('class_name', 'unknown')
        return self._clean_class_name(class_name)
    
    def _clean_class_name(self, class_name: str) -> str:
        """Clean up class names by removing incorrect color prefixes and standardizing names."""
        # Common incorrect color prefixes to remove
        color_prefixes = [
            'a blue', 'blue', 'a red', 'red', 'a green', 'green', 
            'a yellow', 'yellow', 'a white', 'white', 'a black', 'black',
            'a brown', 'brown', 'a grey', 'grey', 'a gray', 'gray',
            'a silver', 'silver', 'a gold', 'gold'
        ]
        
        class_lower = class_name.lower().strip()
        
        # Remove color prefixes if they don't make sense
        for prefix in color_prefixes:
            if class_lower.startswith(prefix + ' '):
                # Keep the color only for specific items where it makes sense
                remaining = class_lower[len(prefix) + 1:]
                # These items can have colors
                color_allowed_items = ['vase', 'pillow', 'cushion', 'blanket', 'rug', 'curtain']
                if not any(item in remaining for item in color_allowed_items):
                    class_name = remaining
                    class_lower = remaining
        
        # Remove unnecessary articles
        if class_lower.startswith('a '):
            class_name = class_name[2:]
        elif class_lower.startswith('an '):
            class_name = class_name[3:]
        elif class_lower.startswith('the '):
            class_name = class_name[4:]
            
        # Standardize common furniture names
        replacements = {
            'couch': 'sofa',
            'television': 'tv',
            'television receiver': 'tv',
            'entertainment center': 'tv stand',
            'coffee table': 'table',
            'dining table': 'table',
            'end table': 'table',
            'side table': 'table',
            'stainless steel refrigerator': 'refrigerator',
            'silver refrigerator': 'refrigerator',
        }
        
        class_lower = class_name.lower().strip()
        for old, new in replacements.items():
            if class_lower == old:
                class_name = new
                break
        
        return class_name.strip()

    def _is_furniture_or_decor(self, class_name: str) -> bool:
        """Check if the class name represents furniture or decor items."""
        class_lower = class_name.lower()
        
        # Define what we consider furniture and decor
        furniture_keywords = {
            # Seating
            'chair', 'couch', 'sofa', 'armchair', 'bench', 'stool', 'ottoman',
            'loveseat', 'recliner', 'sectional',
            # Tables
            'table', 'desk', 'nightstand', 'end table', 'coffee table', 
            'dining table', 'side table', 'console',
            # Storage
            'cabinet', 'shelf', 'bookshelf', 'dresser', 'wardrobe', 'drawer',
            'closet', 'armoire', 'chest', 'hutch', 'buffet', 'sideboard',
            # Bedroom
            'bed', 'mattress', 'headboard', 'nightstand',
            # Decor
            'lamp', 'mirror', 'vase', 'plant', 'cushion', 'pillow', 'rug',
            'carpet', 'curtain', 'blind', 'painting', 'picture', 'frame',
            'clock', 'chandelier', 'light', 'fixture',
            # Electronics
            'tv', 'television', 'monitor', 'screen', 'speaker',
            'entertainment center', 'tv stand',
            # Kitchen/Appliances
            'refrigerator', 'fridge', 'stove', 'oven', 'dishwasher',
            'microwave', 'toaster', 'coffee maker',
            # Other furniture
            'basket', 'bin', 'box', 'trunk', 'stand'
        }
        
        # Exclude these even if they contain furniture keywords
        exclude_keywords = {
            'room', 'floor', 'wall', 'ceiling', 'window', 'door',
            'building', 'house', 'kitchen', 'bathroom', 'bedroom',
            'living room', 'dining room', 'hallway', 'corner',
            'space', 'area', 'background', 'foreground'
        }
        
        # Check exclusions first
        for exclude in exclude_keywords:
            if exclude in class_lower:
                return False
        
        # Check if it's furniture/decor
        for keyword in furniture_keywords:
            if keyword in class_lower:
                return True
        
        return False

    def _extract_contours(self, annotation: Dict, bbox: List[int]) -> List[List[List[int]]]:
        """
        Extracts contours from either RLE or polygon segmentation data.
        """
        segmentation = annotation.get('segmentation')
        if not segmentation:
            return []

        # The RLE mask needs to be decoded relative to the original image size, not the bbox
        img_width, img_height = self.original_size
        
        # The new _rle_to_contours function handles the scaling to the display size correctly
        return self._rle_to_contours(segmentation, (img_width, img_height))

    def run_sam_detection(self, image_path: str, timeout_minutes: int = 5) -> List[Dict]:
        """Run SAM detection using the simpler replicate.run() API."""
        if not self.enabled:
            return []
            
        try:
            # Use the same detect_objects method which now uses replicate.run()
            return self.detect_objects(image_path)
            
        except Exception as e:
            logger.error(f"SAM detection failed: {e}")
            return []

    # ------------------------------------------------------------------
    #                 Replicate utility helpers
    # ------------------------------------------------------------------

    def _get_latest_version_id(self) -> str:
        """Return the latest version id for the configured SAM model slug.

        The result is cached to avoid hitting the Replicate API repeatedly.
        If a fixed version id is provided via env var, that wins.
        """
        if self._fixed_version_id:
            return self._fixed_version_id

        if self._cached_version_id:
            return self._cached_version_id

        try:
            model = replicate.models.get(self._model_slug)
            # versions are returned newest first
            version = model.versions.list()[0]
            self._cached_version_id = version.id
            logger.info(f"ℹ️ Using latest SAM version id: {self._cached_version_id}")
            return self._cached_version_id
        except Exception as e:
            logger.error(f"Failed to fetch latest SAM model version id: {e}")
            # Fallback to hard-coded known version (last known good)
            fallback = "cd9d2617-4bdd-41bb-8059-9ac7b1bfd5ab"
            logger.warning(f"Falling back to hard-coded SAM version id {fallback}")
            return fallback

    def _create_prediction(self, image_data_url: str) -> Optional[str]:
        """Create a prediction on Replicate and return the prediction id.

        Automatically retries once if the version id is invalid (HTTP 422).
        """
        version_id = self._get_latest_version_id()
        try:
            prediction = replicate.predictions.create(
                version=version_id,
                input={
                    "image": image_data_url,  # Changed from "input_image" to "image"
                    "output_json": True      # Request JSON output
                },
            )
            return prediction.id
        except replicate.exceptions.ReplicateError as re:
            # Retry once if it's a 422 (version no longer valid)
            if "422" in str(re):
                logger.warning("Version id invalid, fetching latest and retrying…")
                # Clear cache and retry
                self._cached_version_id = None
                version_id = self._get_latest_version_id()
                prediction = replicate.predictions.create(
                    version=version_id,
                    input={
                        "image": image_data_url,  # Changed from "input_image" to "image"
                        "output_json": True      # Request JSON output
                    },
                )
                return prediction.id
            raise