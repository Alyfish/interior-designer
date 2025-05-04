import os
import sys
import json
import time
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import base64
import uuid

# Import from detection module
from interior_designer.detection.yolo_detector import process_with_yolo, create_yolo_objects
from interior_designer.ui.interactive_ui import create_interactive_html

# Import from analysis module
from interior_designer.analysis.furniture_analyzer import FurnitureAnalyzer
from interior_designer.analysis.perplexity_analyzer import PerplexityImageAnalyzer

class InteriorDesignConnector:
    """
    Connector class that integrates object detection with furniture analysis APIs
    """
    def __init__(self, laozhang_api_key=None, perplexity_api_key=None):
        self.laozhang_api_key = laozhang_api_key
        self.perplexity_api_key = perplexity_api_key
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize analyzers if API keys provided
        self.laozhang_analyzer = None
        self.perplexity_analyzer = None
        
        if laozhang_api_key:
            self.laozhang_analyzer = FurnitureAnalyzer(laozhang_api_key)
        
        if perplexity_api_key:
            self.perplexity_analyzer = PerplexityImageAnalyzer(perplexity_api_key)
    
    def detect_objects(self, image_path, model=None):
        """
        Detect objects in the image using YOLOv8-segmentation
        """
        if model is None:
            from ultralytics import YOLO
            model = YOLO('yolov8x-seg.pt')
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Process with YOLO
        results = process_with_yolo(image, model)
        
        # Create object maps
        objects = create_yolo_objects(results, image)
        
        # Save processed image with segmentation
        output_img_path = os.path.join(self.output_dir, "segmented_image.jpg")
        self._save_segmented_image(image, objects, output_img_path)
        
        return image, objects, output_img_path
    
    def _save_segmented_image(self, image, objects, output_path):
        """
        Save a visualization of the segmented image
        """
        # Make a copy of the image to draw on
        img_copy = image.copy()
        
        # Draw segmentation masks with transparency
        for obj in objects:
            # Extract class name and color
            class_name = obj.get('class', 'unknown')
            color_hex = obj.get('color', '#FF0000')
            
            # Convert hex color to BGR
            color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            
            # Draw contours
            for contour in obj.get('contours', []):
                # Convert contour to numpy array format required by OpenCV
                contour_np = np.array(contour, dtype=np.int32)
                cv2.polylines(img_copy, [contour_np], True, color_bgr, 2)
                
                # Add label text
                if len(contour) > 0:
                    pos = tuple(contour[0])
                    cv2.putText(img_copy, class_name, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, color_bgr, 2)
        
        # Save the result
        cv2.imwrite(output_path, img_copy)
        return output_path
    
    def analyze_furniture(self, image_path, use_perplexity=False):
        """
        Analyze furniture in the image to find matching products
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        
        # Choose the analyzer based on user preference or availability
        analyzer = None
        if use_perplexity and self.perplexity_analyzer:
            analyzer = self.perplexity_analyzer
        elif self.laozhang_analyzer:
            analyzer = self.laozhang_analyzer
        else:
            raise ValueError("No API key provided for furniture analysis")
        
        # Analyze the furniture
        furniture_items = analyzer.analyze_furniture(image_path)
        
        # Save analysis results
        output_text_path = os.path.join(self.output_dir, "furniture_analysis.txt")
        self._save_analysis_results(furniture_items, output_text_path)
        
        # Create JSON output
        output_json_path = os.path.join(self.output_dir, "furniture_analysis.json")
        self._save_analysis_json(furniture_items, output_json_path)
        
        return furniture_items, output_text_path, output_json_path
    
    def _save_analysis_results(self, furniture_items, output_path):
        """
        Save furniture analysis results to a text file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Furniture Analysis Results\n")
            f.write("=========================\n\n")
            
            for item in furniture_items:
                f.write(f"{item.category}:\n")
                f.write(f"{item.description}\n\n")
                
                f.write("Product URLs:\n")
                for url in item.product_urls:
                    status = "[VALID]" if url.is_valid else "[INVALID]"
                    f.write(f"{status} {url.retailer} - {url.price}\n")
                    f.write(f"  {url.url}\n")
                    f.write(f"  {url.description}\n\n")
                
                f.write("-" * 50 + "\n\n")
        
        return output_path
    
    def _save_analysis_json(self, furniture_items, output_path):
        """
        Save furniture analysis results to a JSON file
        """
        # Convert dataclass objects to dictionaries
        items_dict = []
        for item in furniture_items:
            item_dict = {
                "category": item.category,
                "description": item.description,
                "product_urls": []
            }
            
            for url in item.product_urls:
                url_dict = {
                    "url": url.url,
                    "retailer": url.retailer,
                    "price": url.price,
                    "description": url.description,
                    "is_valid": url.is_valid,
                    "error_message": url.error_message
                }
                item_dict["product_urls"].append(url_dict)
            
            items_dict.append(item_dict)
        
        # Save to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items_dict, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def process_image(self, image_path, use_perplexity=False):
        """
        Full processing pipeline: detect objects and then analyze furniture
        """
        # Step 1: Detect objects
        print("Step 1: Detecting objects...")
        image, objects, segmented_image_path = self.detect_objects(image_path)
        
        # Step 2: Save transformed image for API analysis
        print("Step 2: Preparing image for API analysis...")
        transformed_image_path = os.path.join(self.output_dir, "transformed_image.jpg")
        cv2.imwrite(transformed_image_path, image)
        
        # Step 3: Analyze furniture
        print("Step 3: Analyzing furniture...")
        furniture_items, analysis_text_path, analysis_json_path = self.analyze_furniture(
            transformed_image_path, use_perplexity
        )
        
        # Step 4: Return results
        results = {
            "objects": objects,
            "furniture_items": furniture_items,
            "segmented_image_path": segmented_image_path,
            "analysis_text_path": analysis_text_path,
            "analysis_json_path": analysis_json_path
        }
        
        return results

# Main execution for testing
if __name__ == "__main__":
    # Default API keys (should be replaced with actual keys)
    LAOZHANG_API_KEY = "sk-JDMYnZIoNuIHnw560b1a618f3a1c4d2eA7778577012dAeF8"
    PERPLEXITY_API_KEY = "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"
    
    # Create connector
    connector = InteriorDesignConnector(
        laozhang_api_key=LAOZHANG_API_KEY,
        perplexity_api_key=PERPLEXITY_API_KEY
    )
    
    # Process an image
    image_path = "test_img.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Choose API
    use_perplexity = "--perplexity" in sys.argv
    
    # Run the full pipeline
    try:
        print(f"Processing image: {image_path}")
        results = connector.process_image(image_path, use_perplexity)
        
        print("\nProcessing completed successfully!")
        print(f"Segmented image: {results['segmented_image_path']}")
        print(f"Furniture analysis: {results['analysis_text_path']}")
        print(f"JSON results: {results['analysis_json_path']}")
        
        # Print summary of findings
        print("\nDetected objects:")
        object_counts = {}
        for obj in results["objects"]:
            class_name = obj.get("class", "unknown")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}")
        
        print("\nFurniture items with product matches:")
        for item in results["furniture_items"]:
            valid_urls = [url for url in item.product_urls if url.is_valid]
            print(f"  - {item.category}: {len(valid_urls)} product matches")
    
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        print(traceback.format_exc()) 