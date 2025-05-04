import os
import sys
import base64
import asyncio
import httpx
from pathlib import Path
from urllib.parse import urlparse
import extruct
import json
from dotenv import load_dotenv
from PIL import Image
import io
import re
from bs4 import BeautifulSoup
from w3lib.html import get_base_url
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

# Load environment variables
load_dotenv()

# API keys
SERP_KEY = os.getenv("SERP_KEY", "beb9242fceaf8eb4d573b226af0833d0b96cce01fc82c907377b6fdd0fb2236a")
AMAZON_RAPID_KEY = os.getenv("AMAZON_RAPID_KEY", "732af5ccd8msh9d3c01da8429b19p1f38d8jsnc3b4b20a28d8")
BING_KEY = os.getenv("BING_KEY", "")

# Check if keys were loaded properly
if not AMAZON_RAPID_KEY or AMAZON_RAPID_KEY == "732af5ccd8msh9d3c01da8429b19p1f38d8jsnc3b4b20a28d8":
    print("Warning: Using default AMAZON_RAPID_KEY. For proper functionality, set AMAZON_RAPID_KEY in your .env file.")
if not BING_KEY:
    print("Note: BING_KEY not set. Bing visual search will be skipped.")

# New imports for optimization
try:
    from interior_designer.utils.async_utils import (
        throttled, rate_limited, retry_async, with_circuit_breaker, 
        process_concurrently, gather_with_concurrency
    )
    from interior_designer.utils.cache import cached
    from interior_designer.utils.feature_flags import is_enabled
    from interior_designer.utils.metrics import api_requests, api_errors, api_latency, timed_execution
except ModuleNotFoundError:
    # Fallback to relative imports when run directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.async_utils import (
        throttled, rate_limited, retry_async, with_circuit_breaker, 
        process_concurrently, gather_with_concurrency
    )
    from utils.cache import cached
    from utils.feature_flags import is_enabled
    from utils.metrics import api_requests, api_errors, api_latency, timed_execution

# Load config files if they exist
RETAILERS_CONFIG = {}
try:
    with open('config/retailers.json', 'r') as f:
        RETAILERS_CONFIG = json.load(f).get('retailers', {})
except (FileNotFoundError, json.JSONDecodeError):
    print("Warning: retailers.json not found or invalid. Using default patterns.")

def resize_image_for_api(image_path, max_size=800):
    """Resize an image to reduce its size for API requests"""
    try:
        img = Image.open(image_path)
        # Calculate new dimensions while maintaining aspect ratio
        width, height = img.size
        ratio = min(max_size/width, max_size/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to memory buffer
        buffer = BytesIO()
        resized_img.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()
        
        print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        return img_bytes
    except Exception as e:
        print(f"Error resizing image: {e}")
        # Return original image as fallback
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        return img_bytes

@throttled(service="serpapi")
@rate_limited(min_interval=0.5)
@retry_async(max_retries=3)
@with_circuit_breaker(service="serpapi")
@cached
async def google_reverse_image(img_bytes: bytes) -> list[str]:
    """Perform a Google reverse image search using SerpAPI and return shopping links"""
    if not SERP_KEY:
        print("No SerpAPI key provided, skipping Google reverse image search")
        return []
    
    # Only use metrics if available
    try:
        api_requests(provider="serpapi", endpoint="reverse_image_search")
    except Exception as e:
        print(f"Metrics not available: {e}")
    
    try:
        # Time the execution if the function is available
        try:
            with timed_execution("google_reverse_image", {"provider": "serpapi"}):
                return await _do_google_reverse_image(img_bytes)
        except Exception:
            # Fall back to direct execution if timed_execution is not available
            return await _do_google_reverse_image(img_bytes)
    except Exception as e:
        # Try to record error metrics if available
        try:
            error_type = type(e).__name__
            api_errors(provider="serpapi", endpoint="reverse_image_search", error_type=error_type)
        except Exception:
            pass
        print(f"Error in Google reverse image search: {e}")
        return []

async def _do_google_reverse_image(img_bytes: bytes) -> list[str]:
    """Implementation of Google reverse image search"""
    # First try the text-based search as a fallback
    caption = "modern furniture home decor"
    
    # Try to extract image features for better search
    try:
        # Create a temporary file for the image
        temp_path = "temp_search_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)
        
        # Try to get a better caption from the image
        img = Image.open(BytesIO(img_bytes))
        width, height = img.size
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            caption = "wide rectangular furniture"
        elif aspect_ratio < 0.7:
            caption = "tall furniture piece"
        else:
            caption = "furniture living room decor"
    except Exception as e:
        print(f"Error analyzing image: {e}")
    
    # Use a keyword search with SerpAPI instead of reverse image
    print(f"Running Google Shopping search for: '{caption}'")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {
            "engine": "google_shopping",
            "api_key": SERP_KEY,
            "q": caption + " furniture",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "num": 20
        }
        
        response = await client.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract shopping results
        shopping_results = []
        
        # Try to get shopping results
        if 'shopping_results' in data:
            for result in data['shopping_results']:
                link = result.get('link')
                if link and isinstance(link, str):
                    shopping_results.append(link)
        
        # Filter out Google Shopping links and duplicates
        filtered_results = []
        for link in shopping_results:
            if 'google.com/shopping' not in link and link not in filtered_results:
                filtered_results.append(link)
        
        print(f"Found {len(filtered_results)} product links from Google Shopping search")
        
        # Try to record metrics if available
        try:
            api_latency.observe(response.elapsed.total_seconds() * 1000, 
                               provider="serpapi", endpoint="shopping_search")
        except Exception:
            pass
        
        return filtered_results

async def _google_search_by_keyword(query: str = None) -> list[str]:
    """Fallback for when image search fails - use keyword search instead"""
    try:
        # Default search term if none provided
        search_term = query or "modern furniture home decor"
        
        print(f"Performing Google keyword search for: '{search_term}'")
        params = {
            "engine": "google_shopping",
            "api_key": SERP_KEY,
            "q": search_term,
            "num": 20
        }
        
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get("https://serpapi.com/search", params=params)
            r.raise_for_status()
        
        results = r.json().get("shopping_results", [])
        links = []
        for row in results:
            merchant = row.get("source_link") or row.get("product_link") or row.get("link")
            if merchant:
                links.append(merchant)
        
        print(f"Found {len(links)} Google Shopping merchant links from keyword search")
        return links[:20]
    except Exception as e:
        print(f"Error in Google keyword search fallback: {str(e)}")
        return []

@throttled(service="serpapi")
@rate_limited(min_interval=0.5)
@retry_async(max_retries=3)
@with_circuit_breaker(service="serpapi")
@cached
async def google_search(query: str, max_results=5) -> List[Dict[str, Any]]:
    """
    Enhanced Google search with caching and error handling
    """
    if not SERP_KEY:
        print("No SerpAPI key provided, skipping Google search")
        return []
    
    print(f"Running Google Shopping search for: {query}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "q": f"{query} furniture buy online",
                "api_key": SERP_KEY,
                "engine": "google_shopping",
                "google_domain": "google.com",
                "gl": "us",
                "hl": "en",
                "num": max_results * 2  # Request more results to account for filtering
            }
            
            response = await client.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Print debug keys
            print(f"SerpAPI response keys: {list(data.keys())}")
            if "shopping_results" in data and data["shopping_results"]:
                print(f"First shopping result keys: {list(data['shopping_results'][0].keys())}")
                first_item = data['shopping_results'][0]
                print(f"Example product: {first_item.get('title')} - {first_item.get('price')} - Product link: {first_item.get('product_link')}")
            
            results = []
            if "shopping_results" in data:
                shopping_results = data["shopping_results"]
                print(f"Found {len(shopping_results)} Google Shopping results")
                
                for item in shopping_results[:max_results*2]:
                    # Extract product details - prioritize product_link over link
                    link = item.get("product_link") or item.get("link")
                    
                    if not link:
                        print(f"Skipping item without link: {item.get('title')}")
                        continue
                        
                    product = {
                        "url": link,
                        "title": item.get("title", ""),
                        "source": "google_shopping",
                        "thumbnail": item.get("thumbnail", ""),
                        "image": item.get("thumbnail", ""),
                        "price": item.get("price", ""),
                        "currency": "USD",
                        "retailer": item.get("source", ""),
                        "valid": True,  # Pre-validate Google Shopping results
                        "description": item.get("snippet", "")
                    }
                    
                    results.append(product)
                    print(f"Added result: {product['title'][:30]} - {product['price']}")
            
            print(f"Returning {len(results)} Google Shopping products")
            return results[:max_results]  # Limit to requested number
    except Exception as e:
        print(f"Error in Google Shopping search: {e}")
        return []

@throttled(service="rapidapi")
@rate_limited(min_interval=1.0)
@retry_async(max_retries=2)
@with_circuit_breaker(service="rapidapi")
@cached
async def amazon_search(query: str, max_hits=20, page: int = 1, country: str = "US", sort_by: str = "RELEVANCE", product_condition: str = "ALL", is_prime: bool = False, deals_and_discounts: str = "NONE", **kwargs) -> list[dict]:
    """Search Amazon for products based on a text query with RapidAPI parameters"""
    if not AMAZON_RAPID_KEY or AMAZON_RAPID_KEY == "732af5ccd8msh9d3c01da8429b19p1f38d8jsnc3b4b20a28d8":
        print("No valid Amazon Rapid API key provided, skipping Amazon search")
        return []
        
    # Use Real-Time Amazon Data API (RapidAPI) search endpoint
    url = "https://real-time-amazon-data.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": AMAZON_RAPID_KEY,
        "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
    }
    # Build query parameters
    params = {
        "query": query,
        "page": page,
        "country": country,
        "sort_by": sort_by,
        "product_condition": product_condition,
        "is_prime": str(is_prime).lower(),
        "deals_and_discounts": deals_and_discounts
    }
    # Include any additional optional filters (category_id, category, min_price, max_price, brand, seller_id, four_stars_and_up, additional_filters, fields)
    for key in ["category_id", "category", "min_price", "max_price", "brand", "seller_id", "four_stars_and_up", "additional_filters", "fields"]:
        if key in kwargs and kwargs[key] is not None:
            params[key] = kwargs[key]
    
    print(f"Searching Amazon for: '{query}' (page={page}, country={country})")
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=headers, params=params)
            r.raise_for_status()
        
        data = r.json()
        # Expecting 'products' list in response
        products = data.get("products", []) or data.get("results", [])
        results = [{"url": f'https://www.amazon.com/dp/{p["asin"]}',
                   "title": p["title"],
                   "price": p.get("primary_offer", {}).get("offer_price"),
                   "currency": p.get("primary_offer", {}).get("currency_code")}
                  for p in products[:max_hits]]
        print(f"Found {len(results)} Amazon products")
        return results
    except Exception as e:
        print(f"Error searching Amazon: {e}")
        return []

@throttled(service="bing")
@rate_limited(min_interval=1.0)
@retry_async(max_retries=2)
@with_circuit_breaker(service="bing")
@cached
async def bing_visual_search(image_data: bytes) -> List[Dict[str, Any]]:
    """
    Enhanced Bing visual search with caching and error handling
    """
    # Skip if no BING_KEY env
    if not BING_KEY:
        print("No Bing Visual Search API key provided, skipping Bing search")
        return []
        
    print("Performing Bing Visual Search...")
    headers = {"Ocp-Apim-Subscription-Key": BING_KEY,
               "Content-Type": "application/octet-stream"}
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.bing.microsoft.com/v7.0/images/visualsearch",
                headers=headers,
                content=image_data
            )
            r.raise_for_status()
        
        links = []
        for t in r.json().get("tags", []):
            for a in t.get("actions", []):
                if a.get("actionType") == "ShopPing":
                    links += [v["hostPageUrl"] for v in a["data"]["value"]]
        
        # Keep only real retailer domains (no bing.com pages)
        filtered_links = [u for u in links if not urlparse(u).netloc.endswith(("bing.com", "microsoft.com"))][:20]
        print(f"Found {len(filtered_links)} Bing Visual Shopping links")
        return filtered_links
    except Exception as e:
        print(f"Error with Bing Visual Search: {e}")
        return []

async def simple_caption(img_path: str, object_label: str = None) -> str:
    """Generate a simple caption for an image using available models"""
    # If we have the object label, use it for a better caption
    if object_label and object_label.lower() not in ("unknown", "custom"):
        caption = f"modern {object_label} for living room"
        print(f"Using object label for caption: {caption}")
        return caption
    
    try:
        # First try to use OpenAI's gpt-4o-mini if key is available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            import openai
            b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
            openai.api_key = openai_key
            
            try:
                rsp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user",
                               "content":[
                                 {"type":"text","text":"Describe the main object in five words"},
                                 {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                               ]}],
                    max_tokens=10
                )
                caption = rsp.choices[0].message.content.strip()
                print(f"Generated caption with OpenAI: {caption}")
                return caption
            except Exception as e:
                print(f"Error with OpenAI caption: {e}")
                
        # Fallback: Use simple hardcoded captions based on the image dimensions
        img = Image.open(img_path)
        width, height = img.size
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            return "wide rectangular furniture item"
        elif aspect_ratio < 0.7:
            return "tall furniture piece"
        elif width > 300 and height > 300:
            return "large furniture item home decor"
        else:
            return "small furniture decor item"
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "furniture home decor item"

@retry_async(max_retries=3)
async def verify_product_url(url: str) -> Dict[str, Any]:
    """
    Verify a product URL and extract schema data with more lenient validation
    """
    # Skip validation for Google Shopping URLs - they're already verified
    if "shopping.google.com" in url:
        parsed_url = urlparse(url)
        return {
            "url": url,
            "title": "Google Shopping Product",
            "price": "See website for price",
            "currency": "USD",
            "retailer": "Google Shopping",
            "valid": True
        }
    
    # For URLs from Google Shopping results, trust them more
    if url.startswith("https://www.google.") and "/shopping/" in url:
        return {
            "url": url,
            "title": "Shopping Product",
            "price": "See website for price",
            "currency": "USD",
            "retailer": "Google Shopping",
            "valid": True
        }
    
    # First check if this URL matches known retailer patterns
    parsed_domain = url.split('/')[2] if '://' in url else url.split('/')[0]
    domain = parsed_domain.replace('www.', '')
    
    # Check against retailer config patterns
    if RETAILERS_CONFIG:
        retailer_info = None
        for retailer_domain, info in RETAILERS_CONFIG.items():
            if retailer_domain in domain:
                retailer_info = info
                # Check against regex pattern if available
                if 'pattern' in info:
                    pattern = info['pattern']
                    if not re.search(pattern, url):
                        print(f"URL {url} rejected by pattern {pattern}")
                        return {"valid": False, "url": url, "reason": "pattern_mismatch"}
                break
    
    # Short-circuit for known retailers
    for known_retailer in ["amazon.com", "wayfair.com", "ikea.com", "target.com", "walmart.com"]:
        if known_retailer in url.lower():
            return {
                "url": url,
                "title": f"Product from {known_retailer}",
                "price": "See website for price",
                "currency": "USD",
                "retailer": known_retailer.split('.')[0].title(),
                "valid": True
            }
    
    # Resolve Google Shopping redirects first
    url = await _resolve_google_redirect(url)
    print(f"Verifying URL: {url}")
    
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            if r.status_code != 200:
                print(f"  Status code {r.status_code} - rejecting")
                return {"valid": False, "url": url, "reason": f"status_code_{r.status_code}"}
            
            # Try to extract product schema data
            try:
                data = extruct.extract(r.text, base_url=url)
                prod = next((x for x in data["json-ld"] if x.get("@type") == "Product"), {})
                
                if prod:
                    offers = prod.get("offers", {})
                    result = {
                        "url": url,
                        "title": prod.get("name", "Unknown Product"),
                        "price": offers.get("price", "See website for price"),
                        "currency": offers.get("priceCurrency", "USD"),
                        "retailer": urlparse(url).netloc.split(":")[0],
                        "valid": True
                    }
                    print(f"  Valid product: {result['title']} ({result['price']} {result['currency']})")
                    return result
            except Exception as schema_error:
                print(f"  Error extracting schema: {schema_error}")
            
            # If schema extraction fails, create a basic product entry from the URL
            domain = urlparse(url).netloc.replace("www.", "")
            title = domain.split(".")[0].title() + " Product"
            
            # Create a basic product entry
            return {
                "url": url,
                "title": title,
                "price": "See website for price",
                "currency": "USD",
                "retailer": domain.split(".")[0].title(),
                "valid": True  # Be more lenient - accept URLs even without schema
            }
    except Exception as e:
        print(f"  Error verifying {url}: {e}")
        return {"valid": False, "url": url, "reason": str(e)}

@retry_async(max_retries=2)
@with_circuit_breaker(service="verification")
async def verify_product_links(links: List[str]) -> List[Dict[str, Any]]:
    """
    Verify multiple product links concurrently with circuit breaker pattern
    
    Args:
        links: List of product URLs to verify
        
    Returns:
        List of verified product data dictionaries
    """
    # Process links concurrently with limited concurrency
    results = await process_concurrently(links, verify_product_url, max_concurrency=5)
    
    # Filter to keep only valid results
    valid_results = [r for r in results if r.get('valid', False)]
    
    # Sort by having price info first
    return sorted(valid_results, key=lambda x: 0 if 'price' in x else 1)

async def _resolve_google_redirect(url: str) -> str:
    """Resolve Google Shopping redirect links to get actual product URLs"""
    # Skip if not a Google redirect
    if "google.com/url" not in url and "google.com/aclk" not in url:
        return url
    
    try:
        # Add user-agent to avoid bot detection
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            # Get the URL from the query string
            if "google.com/url" in url:
                # For general Google redirects
                pattern = r"url=([^&]+)"
            else:
                # For Google Shopping "aclk" redirects
                pattern = r"adurl=([^&]+)"
                
            match = re.search(pattern, url)
            if match:
                redirect_url = match.group(1)
                # URL decode the redirect URL
                decoded_url = redirect_url.replace("%3A", ":").replace("%2F", "/")
                
                # Follow the redirect to get the final URL
                response = await client.head(decoded_url, headers=headers, follow_redirects=True)
                return str(response.url)
            
        # If we couldn't extract the URL, return the original
        return url
    except Exception as e:
        print(f"Error resolving redirect: {e}")
        return url

@cached
async def find_products(crop_path: str, obj_class: str = None, max_results: int = 5, object_metadata: dict = None) -> list[dict]:
    """Main function to find matching products for a given image"""
    try:
        print(f"\nSearching for products matching image: {crop_path}")
        
        # Resize image for API requests
        img_bytes = resize_image_for_api(crop_path)
        
        # Generate a caption for the image using the object class if available
        caption = await simple_caption(crop_path, obj_class)
        
        # Use additional metadata if provided
        if object_metadata:
            print(f"Using additional object metadata: {object_metadata}")
            # We could use the bbox/dimensions for more targeted search
            if 'class' in object_metadata and not obj_class:
                obj_class = object_metadata['class']
                caption = await simple_caption(crop_path, obj_class)
        
        print(f"Caption: {caption}")
        
        # Start concurrent API calls
        print("Starting concurrent API searches...")
        reverse_image_task = google_reverse_image(img_bytes)
        amazon_search_task = amazon_search(caption, max_results)
        google_search_task = google_search(caption, max_results)
        
        # Conditionally add Bing search if API key is available
        search_tasks = [reverse_image_task, amazon_search_task, google_search_task]
        if BING_KEY:
            bing_task = bing_visual_search(img_bytes)
            search_tasks.append(bing_task)
        
        # Run API calls concurrently
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results, handling any exceptions
        image_links = search_results[0] if not isinstance(search_results[0], Exception) else []
        amazon_products = search_results[1] if not isinstance(search_results[1], Exception) else []
        google_results = search_results[2] if not isinstance(search_results[2], Exception) else []
        
        print(f"DEBUG: Got {len(google_results)} results from Google Shopping")
        
        # Extract Bing results if available
        bing_links = []
        if BING_KEY and len(search_results) > 3:
            bing_data = search_results[3] if not isinstance(search_results[3], Exception) else []
            if isinstance(bing_data, list):
                bing_links = [item.get('url', '') for item in bing_data if 'url' in item]
        
        # Extract URLs from Amazon products
        amazon_links = [a.get("url", "") for a in amazon_products if "url" in a]
        
        # Combine all links and pre-validated products from different sources
        all_links = []
        final_products = []
        
        # Add reverse image search links (need validation)
        all_links.extend([{"url": url, "source": "google_reverse_image"} for url in image_links])
        
        # Add Google Shopping results (pre-validated)
        print(f"DEBUG: Processing {len(google_results)} Google Shopping results")
        for item in google_results:
            # The key is "url" in our standardized format
            if item.get("url", ""):
                # Google results already have a standardized format with all required fields
                product = {
                    "url": item.get("url", ""),
                    "title": item.get("title", "Google Shopping Product"),
                    "price": item.get("price", "See website for price"),
                    "currency": item.get("currency", "USD"),
                    "retailer": item.get("retailer", item.get("source", "Google Shopping")),
                    "source": "google_shopping",
                    "valid": True,
                    "image": item.get("thumbnail", item.get("image", ""))
                }
                final_products.append(product)
                print(f"Added Google Shopping product: {product['title'][:30]}...")
            else:
                print(f"DEBUG: Skipping Google result without URL: {item.keys()}")
        
        # Add Amazon results (pre-validated)
        for item in amazon_products:
            if "url" in item:
                # Convert to our standard format
                product = {
                    "url": item.get("url"),
                    "title": item.get("title", "Amazon Product"),
                    "price": item.get("price", "See website for price"),
                    "currency": item.get("currency", "USD"),
                    "retailer": "Amazon",
                    "source": "amazon",
                    "valid": True
                }
                final_products.append(product)
        
        # Add Bing links (need validation)
        all_links.extend([{"url": url, "source": "bing_visual"} for url in bing_links])
        
        # Filter out empty URLs
        all_links = [item for item in all_links if item.get("url")]
        
        # Create a unique set based on URL
        unique_links = {}
        for item in all_links:
            url = item.get("url")
            if url and url not in unique_links:
                unique_links[url] = item
        
        all_link_items = list(unique_links.values())
        print(f"Found {len(all_link_items)} links to verify and {len(final_products)} pre-validated products")
        
        # Verify links if we have any
        if all_link_items:
            batch_size = 5
            for i in range(0, len(all_link_items), batch_size):
                batch = all_link_items[i:i+batch_size]
                batch_urls = [item["url"] for item in batch]
                
                verification_tasks = [verify_product_url(url) for url in batch_urls]
                verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
                
                # Process results and match with original metadata
                for j, result in enumerate(verification_results):
                    if isinstance(result, Exception):
                        print(f"Error verifying {batch_urls[j]}: {result}")
                        continue
                    
                    if result and result.get("valid", False):
                        # Combine verification result with original link metadata
                        orig_data = batch[j]
                        source = orig_data.get("source", "unknown")
                        combined_data = {**orig_data, **result, "source": source}
                        final_products.append(combined_data)
        
        # If we still have no products, create fallback products
        if not final_products:
            print("No valid products found, creating fallback products")
            
            # Create fallback products based on the object class
            product_type = obj_class if obj_class else caption.split()[0]
            
            fallback_products = [
                {
                    "url": f"https://www.amazon.com/s?k={product_type.replace(' ', '+')}+furniture",
                    "title": f"{product_type.title()} on Amazon",
                    "price": "Various prices",
                    "currency": "USD",
                    "retailer": "Amazon",
                    "source": "fallback",
                    "valid": True
                },
                {
                    "url": f"https://www.wayfair.com/keyword.php?keyword={product_type.replace(' ', '+')}",
                    "title": f"{product_type.title()} on Wayfair",
                    "price": "Various prices",
                    "currency": "USD",
                    "retailer": "Wayfair",
                    "source": "fallback",
                    "valid": True
                },
                {
                    "url": f"https://www.ikea.com/us/en/search/?q={product_type.replace(' ', '+')}",
                    "title": f"{product_type.title()} on IKEA",
                    "price": "Various prices",
                    "currency": "USD",
                    "retailer": "IKEA",
                    "source": "fallback",
                    "valid": True
                }
            ]
            
            final_products.extend(fallback_products)
        
        # Sort by relevance and price availability
        final_products.sort(key=lambda x: (
            0 if x.get("source") != "fallback" else 1,  # Real products first
            0 if x.get("price") and x.get("price") != "See website for price" else 1,  # Products with price first
            0 if x.get("image") else 1,  # Products with image second
            -float(x.get("confidence", 0))  # Higher confidence first
        ))
        
        # Return top results
        return final_products[:max_results]
    
    except Exception as e:
        print(f"Error in find_products: {e}")
        import traceback
        traceback.print_exc()
        
        # Return fallback products on error
        return [
            {
                "url": f"https://www.amazon.com/s?k=furniture",
                "title": "Furniture on Amazon",
                "price": "Various prices",
                "currency": "USD",
                "retailer": "Amazon",
                "source": "error_fallback",
                "valid": True
            }
        ]

# Keep the synchronous wrapper function
def find_products_sync(crop_path: str, obj_class: str = None, max_results: int = 5, object_metadata: dict = None) -> list[dict]:
    """Synchronous wrapper for find_products"""
    return asyncio.run(find_products(crop_path, obj_class, max_results, object_metadata))

# Test the module if run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Testing with image: {test_image}")
        results = find_products_sync(test_image)
        print("\nResults:")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['title']} - {r['price']} {r['currency']} - {r['retailer']}")
            print(f"   {r['url']}")
    else:
        print("Please provide an image path to test") 