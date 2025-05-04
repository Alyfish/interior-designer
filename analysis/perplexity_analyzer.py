import requests
import base64
import json
from typing import Optional, List, Dict
from dataclasses import dataclass
import re
from urllib.parse import urlparse
import time

from .furniture_analyzer import ProductURL, FurnitureItem

class PerplexityImageAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Common shopping domains to verify URLs against
        self.valid_shopping_domains = {
            'amazon.com', 'wayfair.com', 'ikea.com', 'target.com', 
            'walmart.com', 'overstock.com', 'etsy.com', 'homedepot.com',
            'potterybarn.com', 'westelm.com', 'crateandbarrel.com',
            'article.com', 'ashleyfurniture.com', 'livingspaces.com'
        }

    def _is_valid_shopping_url(self, url: str) -> tuple[bool, str]:
        """Check if URL is from a known furniture retailer and well-formed"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # List of valid furniture retailers and their display names
            retailers = {
                'ikea.com': 'IKEA',
                'wayfair.com': 'Wayfair',
                'amazon.com': 'Amazon',
                'target.com': 'Target',
                'walmart.com': 'Walmart',
                'overstock.com': 'Overstock',
                'potterybarn.com': 'Pottery Barn',
                'westelm.com': 'West Elm',
                'crateandbarrel.com': 'Crate & Barrel',
                'article.com': 'Article',
                'ashleyfurniture.com': 'Ashley Furniture',
                'livingspaces.com': 'Living Spaces',
                'allmodern.com': 'AllModern',
                'cb2.com': 'CB2'
            }
            
            # Check if the domain is from a valid retailer
            for retailer_domain, retailer_name in retailers.items():
                if retailer_domain in domain:
                    # Check if it's a product page (contains /p/, /pdp/, /product/, or /products/)
                    if any(x in parsed.path.lower() for x in ['/p/', '/pdp/', '/product/', '/products/']):
                        return True, f"Valid {retailer_name} product URL"
                    else:
                        return False, f"Not a product page on {retailer_name}"
            
            return False, "Not a recognized furniture retailer"
            
        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    def _verify_product_url(self, url: str) -> tuple[bool, str, str]:
        """Verify that a URL is well-formed and from a valid retailer"""
        try:
            is_valid, message = self._is_valid_shopping_url(url)
            return is_valid, message, ""
        except Exception as e:
            return False, f"Error validating URL: {str(e)}", ""

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _extract_price_info(self, description: str) -> tuple[str, str]:
        """Extract price point and style information from description"""
        price_points = {
            'budget': ['budget', 'affordable', 'budget-friendly'],
            'mid': ['mid-range', 'mid range', 'midrange'],
            'premium': ['premium', 'high-end', 'luxury']
        }
        
        price_point = "Unknown"
        style_info = description
        
        # Find the price point mentioned in the description
        lower_desc = description.lower()
        for point, keywords in price_points.items():
            if any(keyword in lower_desc for keyword in keywords):
                price_point = point.title()
                break
        
        return price_point, style_info

    def _parse_furniture_items(self, content: str) -> List[FurnitureItem]:
        """Parse the API response to extract furniture items and URLs"""
        items = {}  # Dictionary to group URLs by category
        current_category = None
        
        # Split content into lines and process each line
        for line in content.split('\n'):
            line = line.strip()
            if not line or any(x in line for x in ['All links verified', '---']):  # Skip separators and summary
                continue
                
            # Check if this is a category line (starts with ** or contains dimensions)
            if line.startswith('**') or ('(' in line and ')' in line and ':' in line):
                # Extract category name before the dimensions
                category = line.split('(')[0].replace('**', '').replace(':', '').strip()
                current_category = category
                if current_category not in items:
                    items[current_category] = {"description": [], "urls": []}
            
            # If we have a current category and this line starts with a retailer name, it's a product
            elif current_category and line.startswith('-') and ' - ' in line:
                # Split the line into retailer, price point, and URL
                parts = line.strip('- ').split(' - ')
                if len(parts) >= 3:
                    retailer = parts[0].strip()
                    price_info = parts[1].strip()
                    url_part = ' - '.join(parts[2:])  # The URL might contain ' - '
                    
                    # Extract URL from the line
                    url_match = re.search(r'(https?://[^\s)]+)', url_part)
                    if url_match:
                        url = url_match.group(1)
                        
                        # Extract price point (Budget, Mid-range, Premium)
                        price_point = "Unknown"
                        if "Budget" in price_info:
                            price_point = "Budget"
                        elif "Mid-range" in price_info:
                            price_point = "Mid"
                        elif "Premium" in price_info:
                            price_point = "Premium"
                        
                        # Get the next line for product details
                        next_line = ""
                        try:
                            next_index = content.split('\n').index(line) + 1
                            next_line = content.split('\n')[next_index].strip()
                            if next_line.startswith('('):
                                next_line = next_line.strip('()')
                        except:
                            pass
                        
                        # Verify URL format and retailer
                        is_valid, error_message, _ = self._verify_product_url(url)
                        
                        # Extract price
                        price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', line)
                        price = price_match.group(0) if price_match else price_point
                        
                        description = f"{retailer} - {price_info}" + (f" - {next_line}" if next_line else "")
                        
                        items[current_category]["urls"].append(ProductURL(
                            url=url,
                            retailer=retailer,
                            price=price,
                            description=description,
                            is_valid=is_valid,
                            error_message=error_message if not is_valid else ""
                        ))
            elif current_category and not line.startswith('-'):
                # This must be a description line
                items[current_category]["description"].append(line)

        # Convert to FurnitureItems
        furniture_items = []
        for category, data in items.items():
            furniture_items.append(FurnitureItem(
                category=category, 
                description='\n'.join(data["description"]),
                product_urls=data["urls"]
            ))
            
        return furniture_items

    def analyze_furniture(self, image_path: str) -> List[FurnitureItem]:
        """
        Analyze furniture in an image and return structured results with shopping links
        """
        try:
            print("Starting image analysis...")
            base64_image = self._encode_image(image_path)
            print("Image encoded successfully")
            image_data_uri = f"data:image/jpeg;base64,{base64_image}"

            system_prompt = """You are a furniture expert and personal shopper. Your task is to analyze the image and find REAL, CURRENTLY AVAILABLE products for purchase. This is ABSOLUTELY CRITICAL:

1. First, analyze each furniture item in detail:
   - Exact material/fabric type
   - Precise color and finish
   - Measured dimensions
   - Key design elements

2. Then, for each item:
   - Go to each retailer's website (IKEA, Wayfair, Target, Amazon)
   - Search for similar items using specific features you identified
   - Find REAL products that are IN STOCK and AVAILABLE NOW
   - Copy the EXACT product URLs from your browser
   - Verify each URL loads a real product page before including it

3. STRICT URL REQUIREMENTS:
   For IKEA:
   - ONLY use URLs in format: ikea.com/us/en/p/[real-product-name]-[8-digit-code]/
   - Example: https://www.ikea.com/us/en/p/soederhamn-sofa-finnsta-white-s69284728/
   
   For Wayfair:
   - ONLY use URLs in format: wayfair.com/furniture/pdp/[product-name]-[letter-number-code].html
   - Example: https://www.wayfair.com/furniture/pdp/steelside-71-wide-sofa-w001494067.html
   
   For Target:
   - ONLY use URLs in format: target.com/p/[product-name]/-/A-[8-digit-code]
   - Example: https://www.target.com/p/carson-sofa-threshold/-/A-54313960
   
   For Amazon:
   - ONLY use URLs in format: amazon.com/dp/[10-character-ASIN]
   - Example: https://www.amazon.com/dp/B07TYCQVFM

4. CRITICAL REQUIREMENTS:
   - ONLY include URLs you have personally verified lead to actual products
   - Include the current price you see on the website
   - Note if the item is in stock
   - NO placeholder or example URLs
   - NO made-up product codes
   - If you can't find a real, available product, say so rather than making one up

5. Response Format:
   [Detailed Item Description with exact measurements]:
   
   - [Retailer] - [Price Point] - [Current Price] - [URL]
   (Include: dimensions, materials, color, in-stock status)"""

            text_prompt = """Please analyze this image and find REAL, CURRENTLY AVAILABLE products that closely match each furniture item. For each piece, search the actual retailer websites and provide working product links in this format:

Example:
Modern Gray Fabric Sofa (measured: 72"W x 35"D x 32"H):

- IKEA - Budget ($499) - IN STOCK
  https://www.ikea.com/us/en/p/soederhamn-sofa-finnsta-white-s69284728/
  (3-seat sofa, light gray fabric, matches dimensions)

- Wayfair - Mid-range ($899) - IN STOCK
  https://www.wayfair.com/furniture/pdp/steelside-71-wide-sofa-w001494067.html
  (71" wide sofa, gray polyester, similar style)

- Target - Premium ($1299) - IN STOCK
  https://www.target.com/p/carson-sofa-threshold/-/A-54313960
  (72" modern sofa, light gray, matches design)

IMPORTANT:
1. Visit each retailer's website and find REAL products
2. Copy actual URLs from your browser
3. Verify each URL loads before including it
4. Include current prices from the website
5. Note stock availability
6. If you can't find a real match, say so instead of making up URLs"""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ]

            payload = {
                "model": "sonar-pro",
                "messages": messages,
                "stream": False,
                "temperature": 0.1  # Lower temperature for more precise responses
            }

            print("Sending request to Perplexity API...")
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            print(f"API Response Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Error Response: {response.text}")
                return []
                
            response.raise_for_status()
            result = response.json()
            
            print("Parsing API response...")
            content = result["choices"][0]["message"]["content"]
            
            items = self._parse_furniture_items(content)
            print(f"\nParsed {len(items)} furniture items")
            return items

        except Exception as e:
            print(f"Error analyzing furniture: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            return [] 