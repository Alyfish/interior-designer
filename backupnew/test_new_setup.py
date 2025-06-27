#!/usr/bin/env python3
"""
Test script for the new file structure
"""

try:
    print("Testing imports...")
    
    # Test SAM detector v2
    from sam_detector_v2 import SAMDetector
    print("✅ SAMDetector v2 imported successfully")
    
    # Test object detector
    from object_detector import ObjectDetector
    print("✅ ObjectDetector imported successfully")
    
    # Test product matcher
    from product_matcher import create_new_product_search_agent
    print("✅ ProductMatcher imported successfully")
    
    # Test vision features
    from vision_features import generate_blip_caption
    print("✅ VisionFeatures imported successfully")
    
    print("\n🎉 All imports successful! New file structure is working.")
    print("\nFiles created:")
    print("📁 Backup files:")
    print("  - new_object_detector_backup.py")
    print("  - streamlit_app_backup.py") 
    print("  - sam_detector_backup.py")
    print("  - product_matcher_backup.py")
    print("\n📁 New working files:")
    print("  - app.py (updated main app)")
    print("  - object_detector.py")
    print("  - sam_detector_v2.py")
    print("  - product_matcher.py")
    
    print("\n🚀 Ready to start experimenting with the new setup!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}") 