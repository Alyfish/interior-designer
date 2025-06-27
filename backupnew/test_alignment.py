#!/usr/bin/env python3
"""
Test script to verify object detection coordinate alignment
Run this to test if the scaling/coordinate fixes are working properly
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from new_object_detector import ObjectDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alignment(image_path: str = None):
    """Test object detection alignment with visualization"""
    
    # Use a test image or prompt for one
    if not image_path:
        # Check for common test image locations
        test_locations = [
            "temp_uploaded_image.jpg",
            "../temp_uploaded_image.jpg",
            "test_image.jpg",
            "sample_room.jpg"
        ]
        
        for loc in test_locations:
            if Path(loc).exists():
                image_path = loc
                break
        
        if not image_path:
            logger.error("No test image found. Please provide an image path.")
            logger.info("Place a test image at 'temp_uploaded_image.jpg' or specify path")
            return
    
    logger.info(f"Testing alignment with image: {image_path}")
    
    try:
        # Initialize detector
        detector = ObjectDetector()
        
        # Get detections
        original_img, objects, segmented_path = detector.detect_objects(image_path)
        
        # Load original image for verification
        orig_height, orig_width = original_img.shape[:2]
        logger.info(f"Original image dimensions: {orig_width}x{orig_height}")
        
        # Create test visualization
        test_img = original_img.copy()
        
        for obj in objects:
            try:
                # Draw bounding box
                if 'bbox' in obj and len(obj['bbox']) == 4:
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # Validate coordinates
                    if x1 < 0 or y1 < 0 or x2 > orig_width or y2 > orig_height:
                        logger.warning(f"Object {obj.get('class', 'unknown')} has out-of-bounds bbox: [{x1}, {y1}, {x2}, {y2}]")
                        continue
                    
                    # Draw bbox
                    color = (0, 255, 0) if obj.get('source') == 'yolo' else (255, 0, 0)
                    cv2.rectangle(test_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{obj['class']} ({obj['source']})"
                    cv2.putText(test_img, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw contours if available
                if 'contours' in obj and obj['contours']:
                    for contour in obj['contours']:
                        if isinstance(contour, list) and len(contour) > 2:
                            # Convert to numpy array format
                            pts = np.array(contour, dtype=np.int32)
                            
                            # Validate contour points
                            valid_pts = []
                            for point in pts:
                                if len(point) >= 2:
                                    x, y = int(point[0]), int(point[1])
                                    if 0 <= x < orig_width and 0 <= y < orig_height:
                                        valid_pts.append([x, y])
                            
                            if len(valid_pts) > 2:
                                valid_pts = np.array(valid_pts, dtype=np.int32)
                                contour_color = (0, 255, 255) if obj.get('source') == 'yolo' else (255, 255, 0)
                                cv2.polylines(test_img, [valid_pts], True, contour_color, 2)
                
            except Exception as e:
                logger.error(f"Error processing object {obj.get('class', 'unknown')}: {e}")
                continue
        
        # Save test result
        output_path = "alignment_test_result.jpg"
        cv2.imwrite(output_path, test_img)
        logger.info(f"✅ Test visualization saved to: {output_path}")
        
        # Print summary
        logger.info(f"📊 Detection Summary:")
        logger.info(f"   • Total objects detected: {len(objects)}")
        
        source_counts = {}
        class_counts = {}
        
        for obj in objects:
            source = obj.get('source', 'unknown')
            class_name = obj.get('class', 'unknown')
            
            source_counts[source] = source_counts.get(source, 0) + 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        logger.info(f"   • By source: {source_counts}")
        logger.info(f"   • By class: {class_counts}")
        
        # Coordinate validation
        coord_issues = 0
        for obj in objects:
            if 'bbox' in obj:
                x1, y1, x2, y2 = obj['bbox']
                if x1 < 0 or y1 < 0 or x2 > orig_width or y2 > orig_height or x1 >= x2 or y1 >= y2:
                    coord_issues += 1
        
        if coord_issues == 0:
            logger.info("✅ All object coordinates are valid and within bounds")
        else:
            logger.warning(f"⚠️  Found {coord_issues} objects with coordinate issues")
        
        return objects, output_path
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return None, None

def test_scaling_factors():
    """Test coordinate scaling calculations"""
    logger.info("🧮 Testing coordinate scaling calculations...")
    
    # Test cases: (original_w, original_h, max_w, max_h, expected_scale)
    test_cases = [
        (800, 600, 1200, 800, 1.0),        # No scaling needed
        (1600, 1200, 1200, 800, 0.666),    # Scale down
        (400, 300, 1200, 800, 1.0),        # No scaling up
        (2400, 1800, 1200, 800, 0.444),    # Large scale down
    ]
    
    for orig_w, orig_h, max_w, max_h, expected in test_cases:
        scale = min(max_w / orig_w, max_h / orig_h, 1)
        display_w = int(orig_w * scale)
        display_h = int(orig_h * scale)
        
        logger.info(f"   {orig_w}x{orig_h} → {display_w}x{display_h} (scale: {scale:.3f})")
        
        if abs(scale - expected) > 0.01:
            logger.warning(f"   ⚠️  Expected scale {expected:.3f}, got {scale:.3f}")
        else:
            logger.info(f"   ✅ Scale calculation correct")

if __name__ == "__main__":
    import sys
    
    logger.info("🔧 Starting Object Detection Alignment Test")
    
    # Test scaling calculations first
    test_scaling_factors()
    
    # Test with provided image or auto-detect
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    objects, output_path = test_alignment(image_path)
    
    if objects and output_path:
        logger.info("🎉 Test completed successfully!")
        logger.info(f"Check the output image: {output_path}")
        logger.info("Look for:")
        logger.info("  • Green boxes = YOLO detections")  
        logger.info("  • Red boxes = SAM detections")
        logger.info("  • Yellow contours = YOLO contours")
        logger.info("  • Cyan contours = SAM contours")
        logger.info("  • All objects should align with actual items in image")
    else:
        logger.error("❌ Test failed - check logs above for details") 