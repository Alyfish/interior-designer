import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

# Import our new product finder
from interior_designer.product_finder import find_products, find_products_sync
from interior_designer.analysis.furniture_analyzer import ProductURL, FurnitureItem

class ProductFinder:
    """
    Integration class that handles cropping objects from images and finding matching products.
    This replaces the LaoZhang and Perplexity API calls for furniture analysis.
    """
    
    def __init__(self, output_dir: str = None):
        """Initialize the product finder"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "output")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def crop_object(self, image: np.ndarray, contours: List[List[int]]) -> Optional[np.ndarray]:
        """Crop an object from an image based on its contours"""
        try:
            # Find the bounding box of the contours
            all_points = np.concatenate([np.array(contour) for contour in contours])
            x_min, y_min = all_points.min(axis=0)
            x_max, y_max = all_points.max(axis=0)
            
            # Add some padding around the object
            height, width = image.shape[:2]
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            # Crop the object
            crop = image[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                print("Crop is empty - object may be too small or at the edge of the image")
                return None
                
            return crop
        except Exception as e:
            print(f"Error cropping object: {e}")
            return None
    
    def find_products_for_object(self, image: np.ndarray, obj: Dict[str, Any]) -> List[ProductURL]:
        """Find products for a single object in the image"""
        try:
            # Get the class and contours from the object
            class_name = obj.get('class', 'unknown')
            contours = obj.get('contours', [])
            
            if not contours:
                print(f"No contours found for {class_name}")
                return self._create_dummy_product_urls(class_name)
                
            # Skip non-furniture objects
            furniture_classes = [
                'chair', 'sofa', 'couch', 'table', 'dining table', 'bed', 
                'desk', 'lamp', 'bookshelf', 'cabinet', 'dresser', 'wardrobe',
                'ottoman', 'stool', 'bench', 'armchair', 'nightstand', 'coffee table',
                'custom'  # Include custom labeled objects
            ]
            
            if class_name.lower() not in [c.lower() for c in furniture_classes]:
                print(f"Skipping non-furniture object: {class_name}")
                return self._create_dummy_product_urls(class_name)
                
            # Crop the object from the image
            crop = self.crop_object(image, contours)
            if crop is None:
                return self._create_dummy_product_urls(class_name)
                
            # Save the cropped image
            timestamp = int(time.time())
            crop_path = os.path.join(self.output_dir, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(crop_path, crop)
            
            # Find products for the cropped image
            print(f"Finding products for {class_name}...")
            # Pass the object class to improve caption quality
            product_results = find_products_sync(crop_path, obj_class=class_name, max_results=5)
            
            # Convert to ProductURL objects
            product_urls = []
            for result in product_results:
                product_urls.append(ProductURL(
                    url=result["url"],
                    retailer=result["retailer"],
                    price=f"{result['price']} {result['currency']}",
                    description=result["title"],
                    is_valid=True
                ))
            
            # If no products found, create a dummy product
            if not product_urls:
                print(f"No products found for {class_name}, creating dummy products")
                product_urls = self._create_dummy_product_urls(class_name)
                
            return product_urls
        except Exception as e:
            print(f"Error finding products for object: {e}")
            import traceback
            traceback.print_exc()
            return self._create_dummy_product_urls(obj.get('class', 'furniture'))
    
    def _create_dummy_product_urls(self, class_name: str) -> List[ProductURL]:
        """Create dummy product URLs when real ones can't be found"""
        retailers = ["Amazon", "Wayfair", "IKEA", "Target"]
        
        product_urls = []
        for i, retailer in enumerate(retailers):
            product_urls.append(ProductURL(
                url=f"https://www.{retailer.lower()}.com/search?q={class_name.replace(' ', '+')}",
                retailer=retailer,
                price=f"${(i+1)*50}.00 USD",
                description=f"{retailer} {class_name.title()} - search for similar products",
                is_valid=True
            ))
        
        return product_urls
    
    def analyze_furniture(self, image_path: str, selected_objects: List[Dict[str, Any]]) -> List[FurnitureItem]:
        """
        Analyze furniture in the image to find matching products.
        This replaces the FurnitureAnalyzer.analyze_furniture method.
        """
        try:
            print(f"Analyzing furniture in {image_path} with {len(selected_objects)} selected objects")
            
            # Validate selected objects
            if not selected_objects:
                print("No objects selected, creating default furniture item")
                return [FurnitureItem(
                    category="Furniture",
                    description="No specific furniture items were selected. Here are some general furniture recommendations.",
                    product_urls=self._create_dummy_product_urls("furniture")
                )]
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Group objects by class
            objects_by_class = {}
            for obj in selected_objects:
                class_name = obj.get('class', 'unknown')
                if class_name not in objects_by_class:
                    objects_by_class[class_name] = []
                objects_by_class[class_name].append(obj)
            
            # Process each class of objects
            furniture_items = []
            for class_name, objects in objects_by_class.items():
                print(f"Processing {len(objects)} {class_name}(s)")
                
                # Find products for all objects of this class
                all_product_urls = []
                for obj in objects:
                    product_urls = self.find_products_for_object(image, obj)
                    all_product_urls.extend(product_urls)
                
                # Deduplicate product URLs by URL
                unique_urls = {}
                for url in all_product_urls:
                    if url.url not in unique_urls:
                        unique_urls[url.url] = url
                unique_product_urls = list(unique_urls.values())
                
                # Create a description
                description = f"{len(objects)} {class_name}(s) detected in the room."
                
                # Create a FurnitureItem
                furniture_items.append(FurnitureItem(
                    category=class_name.capitalize(),
                    description=description,
                    product_urls=unique_product_urls
                ))
            
            # Ensure we return at least one furniture item
            if not furniture_items:
                furniture_items = [FurnitureItem(
                    category="Furniture",
                    description="No furniture items detected. Here are some general furniture recommendations.",
                    product_urls=self._create_dummy_product_urls("furniture")
                )]
            
            return furniture_items
        except Exception as e:
            print(f"Error analyzing furniture: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a default furniture item to avoid UI errors
            return [FurnitureItem(
                category="Furniture",
                description="We encountered an error while analyzing the furniture. Here are some general recommendations instead.",
                product_urls=self._create_dummy_product_urls("furniture")
            )]

# Test the integration if run directly
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Testing with image: {test_image}")
        
        # Create a test object
        test_object = {
            "class": "chair",
            "contours": [[[100, 100], [300, 100], [300, 300], [100, 300]]]
        }
        
        # Create finder and analyze
        finder = ProductFinder()
        results = finder.analyze_furniture(test_image, [test_object])
        
        # Print results
        print("\nResults:")
        for item in results:
            print(f"Category: {item.category}")
            print(f"Description: {item.description}")
            print(f"Product URLs: {len(item.product_urls)}")
            for url in item.product_urls:
                print(f"  - {url.retailer}: {url.price}")
                print(f"    {url.url}")
                print(f"    {url.description}")
    else:
        print("Please provide an image path to test") 