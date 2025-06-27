#!/usr/bin/env python3
"""
Test script to verify SAM detector fixes work correctly
"""
import os
import sys
from PIL import Image, ImageDraw
import tempfile

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam_detector import SAMDetector

def create_test_image(path: str):
    """Create a simple test image with some basic shapes"""
    # Create a 800x600 image with white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some basic shapes that SAM might detect
    # A table (brown rectangle)
    draw.rectangle([100, 400, 300, 500], fill='brown', outline='black', width=2)
    
    # A chair (smaller brown rectangle with back)
    draw.rectangle([350, 450, 450, 500], fill='brown', outline='black', width=2)
    draw.rectangle([350, 400, 370, 450], fill='brown', outline='black', width=2)
    
    # A window (blue rectangle)
    draw.rectangle([500, 100, 700, 250], fill='lightblue', outline='black', width=3)
    
    # Some text
    draw.text((50, 50), "Test Room Image", fill='black')
    
    img.save(path, 'JPEG', quality=85)
    print(f"✅ Created test image: {path}")

def test_sam_detector():
    """Test the SAM detector with comprehensive error handling"""
    print("🔍 Testing SAM Detector...")
    
    # Create test image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        test_image_path = tmp_file.name
    
    try:
        create_test_image(test_image_path)
        
        # Initialize detector
        detector = SAMDetector()
        
        if not detector.enabled:
            print("⚠️ SAM detector is disabled")
            print("Check your REPLICATE_API_TOKEN and ENABLE_SAM_DETECTION settings")
            return
        
        print("🚀 SAM detector enabled, running detection...")
        print("⏳ This may take up to 5 minutes...")
        
        # Run detection
        results = detector.detect_objects(test_image_path)
        
        # Print results
        print(f"✅ Detection completed!")
        print(f"📊 Detected {len(results)} objects")
        
        if results:
            print("\n🏷️ Detected objects:")
            for i, obj in enumerate(results[:10]):  # Show first 10
                class_name = obj.get('class', 'unknown')
                confidence = obj.get('confidence', 0)
                area = obj.get('area', 0)
                proposals = obj.get('class_proposals', [])
                
                print(f"  {i+1}. {class_name} (conf: {confidence:.2f}, area: {area})")
                if proposals:
                    print(f"     Proposals: {', '.join(proposals[:3])}")
        else:
            print("ℹ️ No objects detected")
        
        # Test different image formats
        print("\n🧪 Testing different scenarios...")
        
        # Test with non-existent file
        try:
            detector.detect_objects("nonexistent.jpg")
        except Exception as e:
            print(f"✅ Correctly handled missing file: {type(e).__name__}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"🧹 Cleaned up test image")

if __name__ == "__main__":
    test_sam_detector() 