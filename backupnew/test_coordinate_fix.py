#!/usr/bin/env python3
"""
Test script to verify coordinate alignment fixes for Mask2Former object detection
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json

# Set up environment
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from sam_detector import SAMDetector
from new_object_detector import ObjectDetector

def test_mask2former_detection():
    """Test the fixed Mask2Former detection with coordinate validation"""
    
    print("🧪 Testing Mask2Former Detection Fixes")
    print("=" * 50)
    
    # Initialize detector
    print("📋 Initializing SAM/Mask2Former detector...")
    sam_detector = SAMDetector()
    
    if not sam_detector.enabled:
        print("❌ SAM detector not enabled - check REPLICATE_API_TOKEN")
        return
    
    # Test image
    test_image = "temp_uploaded_image.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🖼️ Testing with image: {test_image}")
    
    # Get image dimensions
    with Image.open(test_image) as img:
        original_width, original_height = img.size
        print(f"📏 Original image dimensions: {original_width}x{original_height}")
    
    # Run detection
    print("🚀 Running Mask2Former detection...")
    detected_objects = sam_detector.run_sam_detection(test_image)
    
    print(f"\n📊 DETECTION RESULTS:")
    print(f"   🔢 Total objects detected: {len(detected_objects)}")
    
    if not detected_objects:
        print("❌ No objects detected!")
        return
    
    # Analyze results
    furniture_count = 0
    non_furniture_count = 0
    bbox_errors = 0
    contour_errors = 0
    
    print(f"\n📋 OBJECT ANALYSIS:")
    for i, obj in enumerate(detected_objects):
        class_name = obj.get('class', 'unknown')
        bbox = obj.get('bbox', [])
        contours = obj.get('contours', [])
        confidence = obj.get('confidence', 0)
        is_furniture = obj.get('is_furniture', False)
        source = obj.get('source', 'unknown')
        
        if is_furniture:
            furniture_count += 1
        else:
            non_furniture_count += 1
        
        # Validate bbox
        bbox_valid = True
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= original_width and 0 <= y1 < y2 <= original_height):
                bbox_valid = False
                bbox_errors += 1
        else:
            bbox_valid = False
            bbox_errors += 1
        
        # Validate contours
        contour_valid = True
        if contours:
            for contour in contours:
                if not isinstance(contour, list) or len(contour) < 3:
                    contour_valid = False
                    contour_errors += 1
                    break
                for point in contour:
                    if not isinstance(point, list) or len(point) != 2:
                        contour_valid = False
                        contour_errors += 1
                        break
                    x, y = point
                    if not (0 <= x <= original_width and 0 <= y <= original_height):
                        contour_valid = False
                        contour_errors += 1
                        break
        
        status = "✅" if (bbox_valid and contour_valid) else "❌"
        furniture_icon = "🛋️" if is_furniture else "📦"
        
        print(f"   {status} {furniture_icon} {i:2d}: {class_name:15s} | "
              f"conf={confidence:.2f} | bbox={bbox_valid} | contours={len(contours):2d} | {source}")
    
    print(f"\n📈 SUMMARY:")
    print(f"   🛋️ Furniture objects: {furniture_count}")
    print(f"   📦 Other objects: {non_furniture_count}")
    print(f"   ❌ BBox errors: {bbox_errors}")
    print(f"   ❌ Contour errors: {contour_errors}")
    
    # Coordinate validation
    print(f"\n🎯 COORDINATE VALIDATION:")
    all_bbox_valid = bbox_errors == 0
    all_contours_valid = contour_errors == 0
    
    print(f"   📍 All bounding boxes valid: {'✅' if all_bbox_valid else '❌'}")
    print(f"   📍 All contours valid: {'✅' if all_contours_valid else '❌'}")
    
    # Create visual test
    print(f"\n🎨 Creating visual test...")
    create_visual_test(test_image, detected_objects, "test_mask2former_alignment.jpg")
    
    return detected_objects

def create_visual_test(image_path, objects, output_path):
    """Create a visual test showing all detected objects with their coordinates"""
    
    # Load image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    print(f"   📏 Creating visualization: {width}x{height}")
    
    # Colors for different object types
    colors = {
        'furniture': (0, 255, 0),      # Green for furniture
        'other': (255, 0, 0),          # Red for other objects
        'bbox': (255, 255, 0),         # Yellow for bboxes
        'contour': (0, 255, 255)       # Cyan for contours
    }
    
    # Draw objects
    for i, obj in enumerate(objects):
        class_name = obj.get('class', 'unknown')
        bbox = obj.get('bbox', [])
        contours = obj.get('contours', [])
        is_furniture = obj.get('is_furniture', False)
        
        color = colors['furniture'] if is_furniture else colors['other']
        
        # Draw bounding box
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), colors['bbox'], 2)
            
            # Add label
            label = f"{i}: {class_name}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['bbox'], 2)
        
        # Draw contours
        if contours:
            for contour in contours:
                if len(contour) >= 3:
                    # Convert to numpy array for cv2
                    contour_np = np.array(contour, dtype=np.int32)
                    cv2.polylines(img, [contour_np], True, color, 2)
    
    # Add legend
    legend_y = 30
    cv2.putText(img, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Green: Furniture contours", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['furniture'], 2)
    cv2.putText(img, "Red: Other object contours", (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['other'], 2)
    cv2.putText(img, "Yellow: Bounding boxes", (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['bbox'], 2)
    
    # Save result
    cv2.imwrite(output_path, img)
    print(f"   💾 Saved visual test: {output_path}")

def test_full_detection_pipeline():
    """Test the complete detection pipeline with both YOLO and SAM"""
    
    print("\n🔧 Testing Full Detection Pipeline")
    print("=" * 50)
    
    detector = ObjectDetector()
    test_image = "temp_uploaded_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🖼️ Testing with image: {test_image}")
    
    # Run full detection
    print("🚀 Running full object detection...")
    try:
        original_image_cv, detected_objects, segmented_image_path = detector.detect_objects(test_image)
        
        print(f"\n📊 FULL PIPELINE RESULTS:")
        print(f"   🔢 Total objects detected: {len(detected_objects)}")
        
        # Count by source
        sources = {}
        for obj in detected_objects:
            source = obj.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        for source, count in sources.items():
            print(f"   📦 {source}: {count} objects")
        
        # Create combined visualization
        if detected_objects:
            create_visual_test(test_image, detected_objects, "test_full_pipeline_alignment.jpg")
        
        return detected_objects
        
    except Exception as e:
        print(f"❌ Error in full pipeline: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("🧪 COORDINATE ALIGNMENT TEST SUITE")
    print("=" * 60)
    
    # Test 1: Mask2Former detection only
    print("\n🟦 TEST 1: Mask2Former Detection")
    mask2former_objects = test_mask2former_detection()
    
    # Test 2: Full pipeline
    print("\n🟦 TEST 2: Full Detection Pipeline")
    full_pipeline_objects = test_full_detection_pipeline()
    
    # Summary
    print(f"\n📋 FINAL SUMMARY:")
    print(f"   🎭 Mask2Former objects: {len(mask2former_objects) if mask2former_objects else 0}")
    print(f"   🎯 Full pipeline objects: {len(full_pipeline_objects) if full_pipeline_objects else 0}")
    
    if mask2former_objects and len(mask2former_objects) > 0:
        print("   ✅ Mask2Former detection working!")
    else:
        print("   ❌ Mask2Former detection failed!")
    
    if full_pipeline_objects and len(full_pipeline_objects) > 0:
        print("   ✅ Full pipeline working!")
    else:
        print("   ❌ Full pipeline failed!")
    
    print("\n🎨 Check the generated images:")
    print("   - test_mask2former_alignment.jpg")
    print("   - test_full_pipeline_alignment.jpg")
    print("\n✅ Test complete!") 