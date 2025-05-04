import requests
import base64
import json
import re
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from urllib.parse import urlparse
import time

# API endpoint (chat completions endpoint is used for both text and multimodal inputs)
API_URL = "https://api.laozhang.ai/v1/chat/completions"

@dataclass
class ProductURL:
    url: str
    retailer: str
    price: str
    description: str
    is_valid: bool = False
    error_message: str = ""

@dataclass
class FurnitureItem:
    category: str
    description: str
    product_urls: List[ProductURL]

class FurnitureAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = API_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Valid retailers and their URL patterns
        self.retailers = {
            'ikea.com': {
                'name': 'IKEA',
                'pattern': r'ikea\.com/us/en/p/[a-z0-9-]+-[a-z0-9]+/?$'
            },
            'wayfair.com': {
                'name': 'Wayfair',
                'pattern': r'wayfair\.com/furniture/pdp/[a-z0-9-]+-[a-z0-9]+\.html$'
            },
            'target.com': {
                'name': 'Target',
                'pattern': r'target\.com/p/[a-z0-9-]+/-/A-\d+$'
            },
            'amazon.com': {
                'name': 'Amazon',
                'pattern': r'amazon\.com/[^/]+/dp/[A-Z0-9]{10}/?$'
            },
            'walmart.com': {
                'name': 'Walmart',
                'pattern': r'walmart\.com/ip/[a-z0-9-]+/\d+$'
            }
        }

    def encode_image(self, image_path: str) -> str:
        """Convert an image to base64 encoding"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def validate_url(self, url: str) -> tuple[bool, str, str]:
        """Validate if a URL matches the expected pattern for furniture retailers"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check if it's from a valid retailer
            for retailer_domain, info in self.retailers.items():
                if retailer_domain in domain:
                    # Check if URL matches the expected pattern
                    if re.match(info['pattern'], url.lower()):
                        return True, info['name'], "Valid product URL"
                    else:
                        return False, info['name'], f"Invalid URL format for {info['name']}"
            
            return False, "Unknown", "Not a recognized furniture retailer"
            
        except Exception as e:
            return False, "Unknown", f"Invalid URL format: {str(e)}"

    def analyze_furniture(self, image_path: str) -> List[FurnitureItem]:
        """Analyze furniture in an image and return structured results with shopping links"""
        try:
            print("Starting image analysis...")
            base64_image = self.encode_image(image_path)
            print("Image encoded successfully")

            system_prompt = """You are a furniture expert and personal shopper. Your task is to analyze the image and find REAL, CURRENTLY AVAILABLE products for purchase. This is ABSOLUTELY CRITICAL:

1. First, analyze each furniture item in detail:
   - Exact material/fabric type
   - Precise color and finish
   - Measured dimensions
   - Key design elements

2. Then, for each item:
   - Go to each retailer's website (IKEA, Wayfair, Target, Amazon, Walmart)
   - Search for similar items using specific features you identified
   - Find REAL products that are IN STOCK and AVAILABLE NOW
   - Copy the EXACT product URLs from your browser
   - Verify each URL loads a real product page before including it

3. STRICT URL REQUIREMENTS:
   For IKEA:
   - ONLY use URLs in format: ikea.com/us/en/p/[product-name]-[product-id]/
   
   For Wayfair:
   - ONLY use URLs in format: wayfair.com/furniture/pdp/[product-name]-[product-id].html
   
   For Target:
   - ONLY use URLs in format: target.com/p/[product-name]/-/A-[product-id]
   
   For Amazon:
   - ONLY use URLs in format: amazon.com/[product-name]/dp/[ASIN]
   
   For Walmart:
   - ONLY use URLs in format: walmart.com/ip/[product-name]/[product-id]

4. Response Format:
   [Detailed Item Description with exact measurements]:
   
   - [Retailer] - [Price Point] - [Current Price]
   [URL]
   (Include: dimensions, materials, color, in-stock status)"""

            payload = {
                "model": "gpt-4o",  # Using the base model for image analysis
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI that analyzes furniture in images and provides detailed product recommendations. DO NOT generate new images - only analyze the provided image."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze this image and find REAL, CURRENTLY AVAILABLE furniture products that match each item. For each item, provide:\n1. Exact material/fabric type\n2. Precise color and finish\n3. Measured dimensions\n4. Key design elements\n5. Links to similar products from IKEA, Wayfair, Target, Amazon, or Walmart that are currently in stock"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,  # Lower temperature for more precise responses
                "max_tokens": 4096,
                "stream": False  # Ensure we get a complete response
            }

            print("Sending request to API...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120  # Increased timeout to 2 minutes for image processing
            )
            print(f"API Response Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Error Response: {response.text}")
                return []
                
            result = response.json()
            print(f"API Response Received")
            content = result["choices"][0]["message"]["content"]
            
            # Parse the response into structured data
            furniture_items = []
            current_item = None
            current_description = []
            current_urls = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a new item description
                if line.endswith(':') and not line.startswith('-'):
                    # Save previous item if exists
                    if current_item and current_urls:
                        furniture_items.append(FurnitureItem(
                            category=current_item,
                            description='\n'.join(current_description),
                            product_urls=current_urls
                        ))
                    
                    # Start new item
                    current_item = line[:-1].strip()
                    current_description = []
                    current_urls = []
                    
                # If line contains a URL, process it
                elif 'http' in line:
                    url_match = re.search(r'(https?://[^\s)]+)', line)
                    if url_match:
                        url = url_match.group(1)
                        is_valid, retailer, message = self.validate_url(url)
                        
                        # Extract price and description
                        price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', line)
                        price = price_match.group(0) if price_match else "Price not found"
                        
                        description = line.replace(url, '').strip()
                        if price in description:
                            description = description.replace(price, '').strip()
                        
                        current_urls.append(ProductURL(
                            url=url,
                            retailer=retailer,
                            price=price,
                            description=description,
                            is_valid=is_valid,
                            error_message=message
                        ))
                else:
                    current_description.append(line)
            
            # Add the last item
            if current_item and current_urls:
                furniture_items.append(FurnitureItem(
                    category=current_item,
                    description='\n'.join(current_description),
                    product_urls=current_urls
                ))
            
            return furniture_items

        except Exception as e:
            print(f"Error analyzing furniture: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            return [] 