# SAM detector with Mask2Former backend for compatibility
import logging
import os
import time
import threading
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from PIL import Image
import replicate

# Force reload environment variables from multiple sources
from dotenv import load_dotenv
import os

# Try multiple .env file locations and encodings
env_loaded = False
for env_file in ['.env', 'env', '.env.local', '../.env']:
    for encoding in ['utf-16le', 'utf-8', 'utf-16', 'cp1252']:
        try:
            if os.path.exists(env_file):
                if load_dotenv(env_file, override=True, encoding=encoding):
                    print(f"✅ Loaded env from {env_file} with {encoding}")
                    env_loaded = True
                    break
        except Exception as e:
            continue
    if env_loaded:
        break

# Also try system environment variables directly
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')

# If still empty, try importing from config
if not REPLICATE_API_TOKEN:
    try:
        from config import REPLICATE_API_TOKEN
    except ImportError:
        REPLICATE_API_TOKEN = ''

logger = logging.getLogger(__name__)

class SAMDetector:
    """
    Mask2Former detector with SAM-compatible interface
    This is a drop-in replacement that uses Mask2Former but keeps the same method names
    """
    
    def __init__(self):
        """Initialize Mask2Former detector with SAM interface"""
        # Check if we should use SAM or Mask2Former
        use_mask2former = os.getenv("USE_MASK2FORMER", "true").lower() in ["true", "on", "yes", "1"]
        
        self.enabled = bool(REPLICATE_API_TOKEN) and os.getenv("ENABLE_SAM_DETECTION", "on").lower() in ["on", "true", "1", "yes"]
        self.model_name = "Mask2Former" if use_mask2former else "SAM"
        
        if self.enabled:
            # Ensure Replicate API token is set in environment
            if REPLICATE_API_TOKEN:
                print(f"🔧 REPLICATE API SETUP:")
                print(f"   🔑 Token: {REPLICATE_API_TOKEN[:15]}...{REPLICATE_API_TOKEN[-5:]} (length: {len(REPLICATE_API_TOKEN)})")
                print(f"   🤖 Model: {self.model_name}")
                print(f"   ✅ Status: ENABLED")
                
                os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
                # Also set it for the replicate client directly
                import replicate
                replicate.Client(api_token=REPLICATE_API_TOKEN)
                logger.info(f"✅ {self.model_name} detector initialized with Replicate API")
                logger.info(f"🔑 Using API token: {REPLICATE_API_TOKEN[:8]}...")
            else:
                print(f"❌ REPLICATE API SETUP FAILED:")
                print(f"   🚨 REPLICATE_API_TOKEN is empty or None")
                print(f"   🤖 Model: {self.model_name}")
                print(f"   ❌ Status: DISABLED")
                logger.error(f"❌ REPLICATE_API_TOKEN is empty or None")
                self.enabled = False
        else:
            print(f"⚠️ REPLICATE API DISABLED:")
            print(f"   🔑 Token available: {'Yes' if REPLICATE_API_TOKEN else 'No'}")
            print(f"   ⚙️ ENABLE_SAM_DETECTION: {os.getenv('ENABLE_SAM_DETECTION', 'not set')}")
            print(f"   🤖 Model: {self.model_name}")
            print(f"   ❌ Status: DISABLED")
            logger.warning(f"⚠️ {self.model_name} detector disabled (check REPLICATE_API_TOKEN and ENABLE_SAM_DETECTION)")
        
        self._cache = {}
    
    def run_sam_detection(self, image_path: str, timeout_minutes: int = 5) -> List[Dict[str, Any]]:
        """
        Run detection using Mask2Former (15x faster than SAM!)
        Keeps the method name for compatibility
        """
        if not self.enabled:
            logger.info(f"{self.model_name} detection skipped - not enabled")
            return []
        
        # Check cache
        if image_path in self._cache:
            logger.info(f"Using cached {self.model_name} results")
            return self._cache[image_path]
        
        try:
            # Prepare image and get scaling info
            image_data_url, scaling_info = self._prepare_image_for_replicate(image_path)
            
            logger.info(f"Running {self.model_name} detection...")
            logger.info(f"🔑 Token check: {REPLICATE_API_TOKEN[:10]}... (length: {len(REPLICATE_API_TOKEN)})")
            start_time = time.time()
            
            # Run Mask2Former (much faster!) - Using latest version from API reference
            try:
                # Create client with explicit API token
                client = replicate.Client(api_token=REPLICATE_API_TOKEN)
                print(f"🚀 REPLICATE API CALL STARTING...")
                print(f"   📧 Model: hassamdevsy/mask2former")
                print(f"   🔑 Token: {REPLICATE_API_TOKEN[:15]}...{REPLICATE_API_TOKEN[-5:]} (length: {len(REPLICATE_API_TOKEN)})")
                print(f"   📂 Image path: {image_path}")
                print(f"   🖼️ Image data URL length: {len(image_data_url)} chars")
                print(f"   📏 Scaling info: {scaling_info}")
                logger.info(f"🚀 Making API call to Mask2Former with token: {REPLICATE_API_TOKEN[:10]}...")
                
                output = client.run(
                    "hassamdevsy/mask2former:86aa30aafd3ade4153ae74aae3e40642a3dff824ed622ff86cec9d67ceb178d2",
                    input={
                        "image": image_data_url
                    }
                )
                
                print(f"✅ REPLICATE API CALL COMPLETED!")
                print(f"   📦 Raw output type: {type(output)}")
                print(f"   📦 Output keys: {list(output.keys()) if output and hasattr(output, 'keys') else 'None/No keys'}")
                if output:
                    print(f"   📄 Full output: {output}")
                
                logger.info(f"✅ API call successful! Output keys: {list(output.keys()) if output else 'None'}")
                if output and 'objects' in output:
                    print(f"   📦 Found {len(output['objects'])} objects in response")
                    logger.info(f"📦 Found {len(output['objects'])} objects in response")
                    
            except Exception as api_error:
                print(f"❌ REPLICATE API CALL FAILED!")
                print(f"   🚨 Error type: {type(api_error).__name__}")
                print(f"   🚨 Error message: {str(api_error)}")
                print(f"   🔑 Token being used: {REPLICATE_API_TOKEN[:15]}...{REPLICATE_API_TOKEN[-5:]} (length: {len(REPLICATE_API_TOKEN)})")
                print(f"   📂 Image path: {image_path}")
                print(f"   🖼️ Image data length: {len(image_data_url)} chars")
                logger.error(f"❌ API call failed: {api_error}")
                logger.error(f"🔑 Token being used: {REPLICATE_API_TOKEN[:15]}... (length: {len(REPLICATE_API_TOKEN)})")
                raise
            
            elapsed = time.time() - start_time
            logger.info(f"✅ {self.model_name} completed in {elapsed:.1f} seconds!")
            
            # Process with scaling info
            detected_objects = self._process_mask2former_output(output, image_path, scaling_info)
            
            # Cache results
            self._cache[image_path] = detected_objects
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"{self.model_name} detection failed: {e}", exc_info=True)
            return []
    
    # Alias for compatibility
    detect_objects = run_sam_detection
    
    def _prepare_image_for_replicate(self, image_path: str) -> tuple[str, Dict[str, Any]]:
        """
        Prepare image for Replicate API and track scaling info
        Returns: (data_url, scaling_info)
        """
        import base64
        from io import BytesIO
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
            
            # Store original dimensions
            original_width, original_height = image.size
            
            # Resize for API (max 1024px)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            else:
                new_width, new_height = original_width, original_height
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            scaling_info = {
                'original_width': original_width,
                'original_height': original_height,
                'processed_width': new_width,
                'processed_height': new_height,
                'scale_x': original_width / new_width,
                'scale_y': original_height / new_height
            }
            
            return f"data:image/png;base64,{img_str}", scaling_info
            
        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            raise
    
    def _process_mask2former_output(self, output: Dict[str, Any], original_image_path: str, scaling_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process Mask2Former output with proper coordinate mapping and scaling"""
        detected_objects = []
        
        if not output:
            logger.warning("Empty Mask2Former output")
            return []
            
        # Get original image dimensions
        original_width = scaling_info['original_width']
        original_height = scaling_info['original_height']
        processed_width = scaling_info['processed_width']
        processed_height = scaling_info['processed_height']
        
        print(f"🔧 COORDINATE PROCESSING:")
        print(f"   📏 Original size: {original_width}x{original_height}")
        print(f"   📏 Processed size: {processed_width}x{processed_height}")
        print(f"   📐 Scale factors: {original_width/processed_width:.3f}x, {original_height/processed_height:.3f}y")
        
        # Handle Mask2Former output format: {'objects': [...], 'segment': <FileOutput>}
        if not isinstance(output, dict) or 'objects' not in output or 'segment' not in output:
            logger.error(f"Invalid Mask2Former output format. Expected dict with 'objects' and 'segment' keys. Got: {type(output)} with keys: {list(output.keys()) if isinstance(output, dict) else 'N/A'}")
            return []
            
        objects_list = output['objects']
        segment_url = output['segment']
        
        print(f"   📦 Processing {len(objects_list)} objects from Mask2Former")
        print(f"   🖼️ Segmentation mask URL: {segment_url}")
        
        # Download the segmentation mask
        try:
            mask_image = self._download_image(str(segment_url))
            print(f"   📥 Downloaded mask image: {mask_image.shape}")
            mask_height, mask_width = mask_image.shape[:2]
        except Exception as e:
            logger.error(f"Failed to download segmentation mask: {e}")
            return []
        
        # Calculate scaling factors from mask to original image
        scale_x = original_width / mask_width
        scale_y = original_height / mask_height
        print(f"   📐 Mask->Original scale factors: {scale_x:.3f}x, {scale_y:.3f}y")
        
        # Process ALL objects first (don't filter furniture early)
        object_id = 0
        for i, obj_info in enumerate(objects_list):
            try:
                color = obj_info['color']  # [R, G, B]
                label = obj_info['label']
                
                print(f"   📦 Processing object {i}: {label} with color {color}")
                
                # Extract mask for this object color
                object_mask = self._extract_color_mask(mask_image, color)
                if object_mask is None:
                    print(f"      ❌ No mask found for {label}")
                    continue
                
                # Find contours with proper scaling
                contours, bbox = self._mask_to_contours_scaled(object_mask, scale_x, scale_y)
                
                if not contours or not bbox:
                    print(f"      ❌ No valid contours for {label}")
                    continue
                
                # Calculate confidence based on area and object type
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                img_area = original_width * original_height
                area_ratio = bbox_area / img_area
                
                # Higher confidence for furniture/decor
                is_furniture = self._is_furniture_or_decor(label)
                if is_furniture:
                    confidence = min(0.99, 0.85 + area_ratio * 10)
                else:
                    confidence = min(0.95, 0.70 + area_ratio * 10)
                
                # Create object entry (include ALL objects, filter later in UI if needed)
                obj = {
                    'id': f'mask2former_{object_id}',
                    'class': label,
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'contours': contours,
                    'area': int(bbox_area),
                    'source': 'mask2former',
                    'color': color,
                    'original_index': i,
                    'is_furniture': is_furniture
                }
                
                detected_objects.append(obj)
                object_id += 1
                
                print(f"      ✅ Added {label}: bbox={bbox}, area={bbox_area:.0f}, confidence={confidence:.2%}, furniture={is_furniture}")
                
            except Exception as e:
                logger.error(f"Error processing object {i} ({obj_info.get('label', 'unknown')}): {e}")
                continue
        
        print(f"✅ Successfully processed {len(detected_objects)} objects from Mask2Former")
        return detected_objects
    
    def _extract_color_mask(self, mask_image: np.ndarray, target_color: List[int], tolerance: int = 5) -> Optional[np.ndarray]:
        """Extract binary mask for specific color with improved color matching"""
        if len(mask_image.shape) != 3:
            return None
            
        # Convert RGB to BGR for OpenCV
        target_bgr = [target_color[2], target_color[1], target_color[0]]
        
        # Create color bounds with tolerance
        lower = np.array([max(0, c - tolerance) for c in target_bgr])
        upper = np.array([min(255, c + tolerance) for c in target_bgr])
        
        # Create mask
        mask = cv2.inRange(mask_image, lower, upper)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Check if mask has enough pixels
        if np.sum(mask) > 50:  # Lower threshold
            return mask
        
        return None
    
    def _mask_to_contours_scaled(self, binary_mask: np.ndarray, scale_x: float, scale_y: float) -> tuple[List[List[List[int]]], List[int]]:
        """Convert binary mask to scaled contours with proper coordinate handling"""
        # Find contours
        contours_cv, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours_cv:
            return [], []
        
        # Process all significant contours
        all_contours = []
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for contour in contours_cv:
            # Skip very small contours (noise)
            area = cv2.contourArea(contour)
            if area < 50:  # Lower threshold for Mask2Former
                continue
            
            # Scale contour points to original image coordinates
            scaled_contour = []
            for point in contour.reshape(-1, 2):
                scaled_x = int(point[0] * scale_x)
                scaled_y = int(point[1] * scale_y)
                
                # Ensure within image bounds
                scaled_x = max(0, min(scaled_x, int(scale_x * binary_mask.shape[1])))
                scaled_y = max(0, min(scaled_y, int(scale_y * binary_mask.shape[0])))
                
                scaled_contour.append([scaled_x, scaled_y])
                
                # Update bounding box
                min_x = min(min_x, scaled_x)
                min_y = min(min_y, scaled_y)
                max_x = max(max_x, scaled_x)
                max_y = max(max_y, scaled_y)
            
            if len(scaled_contour) > 2:  # Valid contour
                all_contours.append(scaled_contour)
        
        if not all_contours:
            return [], []
        
        # Create bounding box
        bbox = [int(min_x), int(min_y), int(max_x), int(max_y)]
        
        return all_contours, bbox
    
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
        
        # Define what we consider furniture and decor (expanded for Mask2Former outputs)
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
            'microwave', 'toaster', 'coffee maker', 'coffee',
            # Other furniture and items
            'basket', 'bin', 'box', 'trunk', 'stand', 'pot', 'tray', 'plate',
            'glass', 'bottle', 'book', 'flower', 'blanket', 'door'
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
        
    def _segmentation_to_contours(self, segmentation, width, height, scale_x, scale_y):
        """Convert segmentation data to contours with scaling"""
        contours = []
        
        if isinstance(segmentation, dict) and 'counts' in segmentation:
            # RLE format
            try:
                from pycocotools import mask as mask_utils
                binary_mask = mask_utils.decode(segmentation)
                
                # Scale mask if needed
                if binary_mask.shape != (height, width):
                    binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Find contours
                contours_cv, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours_cv:
                    points = contour.reshape(-1, 2).tolist()
                    if len(points) > 2:
                        contours.append(points)
            except ImportError:
                logger.warning("pycocotools not available for RLE decoding")
                
        elif isinstance(segmentation, list):
            # Polygon format
            for seg in segmentation:
                if isinstance(seg, list) and len(seg) > 4:
                    # Convert flat list to points and scale
                    points = []
                    for i in range(0, len(seg), 2):
                        if i + 1 < len(seg):
                            x = int(seg[i] * scale_x)
                            y = int(seg[i + 1] * scale_y)
                            # Ensure within bounds
                            x = max(0, min(x, width))
                            y = max(0, min(y, height))
                            points.append([x, y])
                    if len(points) > 2:
                        contours.append(points)
        
        return contours

    def _bbox_to_contour(self, bbox):
        """Convert bbox to contour points"""
        x1, y1, x2, y2 = bbox
        return [
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ]
    
    def _download_image(self, url: str) -> np.ndarray:
        """Download image from URL"""
        import requests
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    

    
    def clear_cache(self):
        """Clear detection cache"""
        self._cache.clear()
        logger.info(f"{self.model_name} cache cleared") 