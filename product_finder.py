import os
import sys
import base64
import asyncio
import httpx
from pathlib import Path
from urllib.parse import urlparse, urljoin
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
    
    # Skip if feature flag is disabled
    if not is_enabled("SERPAPI_REVIMG"):
        print("SerpAPI reverse image search is disabled by feature flag")
        return await _google_search_by_keyword(caption="furniture item")
    
    try:
        print("Performing SerpAPI reverse image search with direct file upload")
        
        # Time the execution if the function is available
        try:
            with timed_execution("google_reverse_image", {"provider": "serpapi"}):
                return await _do_serpapi_reverse_image(img_bytes)
        except Exception:
            # Fall back to direct execution if timed_execution is not available
            return await _do_serpapi_reverse_image(img_bytes)
    except Exception as e:
        # Try to record error metrics if available
        try:
            error_type = type(e).__name__
            api_errors(provider="serpapi", endpoint="reverse_image_search", error_type=error_type)
        except Exception:
            pass
        print(f"Error in SerpAPI reverse image search: {e}")
        
        # Fall back to keyword-based search
        print("Falling back to keyword-based search")
        return await _google_search_by_keyword()

async def _do_serpapi_reverse_image(img_bytes: bytes) -> list[str]:
    """
    Implementation of SerpAPI reverse image search using direct file upload.
    This gives better results than the previous text-based search approach.
    """
    try:
        # Use multipart/form-data request with image file
        url = "https://serpapi.com/search"
        
        # Create temporary file for the image
        temp_path = "temp_search_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)
        
        # Prepare the request using httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            # First try direct reverse image search with file upload
            params = {
                "engine": "google_lens",
                "api_key": SERP_KEY,
                "google_domain": "google.com",
                "hl": "en"
            }
            
            # Read the file in binary mode
            with open(temp_path, "rb") as f:
                files = {"image_file": ("image.jpg", f, "image/jpeg")}
                
                # Post request with image file
                print("Sending image to SerpAPI Google Lens API...")
                response = await client.post(url, params=params, files=files)
                response.raise_for_status()
                data = response.json()
                
                # Extract product links from the response
                product_links = []
                
                # Check for visual matches
                if "visual_matches" in data:
                    print(f"Found {len(data['visual_matches'])} visual matches")
                    for item in data["visual_matches"]:
                        if "link" in item and item["link"].startswith("http"):
                            product_links.append(item["link"])
                
                # Extract shopping results
                if "shopping_results" in data:
                    print(f"Found {len(data['shopping_results'])} shopping results from image search")
                    for item in data["shopping_results"]:
                        if "link" in item and item["link"].startswith("http"):
                            product_links.append(item["link"])
                            
                        # Also check for product_link which is a more specific field
                        if "product_link" in item and item["product_link"].startswith("http"):
                            product_links.append(item["product_link"])
                
                # Try a secondary search: Google Shopping search
                if len(product_links) < 2 and "google_related_searches" in data:
                    related_terms = [item.get("query", "") for item in data.get("google_related_searches", [])]
                    
                    if related_terms:
                        best_term = related_terms[0]  # Use the most relevant term
                        print(f"Using related search term from image: '{best_term}'")
                        
                        # Run a Google Shopping search with the related term
                        shop_params = {
                            "engine": "google_shopping",
                            "api_key": SERP_KEY,
                            "q": f"{best_term} buy",
                            "google_domain": "google.com",
                            "gl": "us",
                            "hl": "en"
                        }
                        
                        shop_response = await client.get(url, params=shop_params)
                        if shop_response.status_code == 200:
                            shop_data = shop_response.json()
                            
                            if "shopping_results" in shop_data:
                                print(f"Found {len(shop_data['shopping_results'])} shopping results from keyword")
                                for item in shop_data["shopping_results"]:
                                    if "product_link" in item and item["product_link"].startswith("http"):
                                        product_links.append(item["product_link"])
                
                # Remove duplicates
                unique_links = []
                for link in product_links:
                    if link not in unique_links:
                        unique_links.append(link)
                
                if unique_links:
                    print(f"Found {len(unique_links)} product links from reverse image search")
                    return unique_links[:20]  # Return top 20 links
        
        # If we get here with no results, fall back to caption-based search
        return await _google_search_by_keyword()
        
    except Exception as e:
        print(f"Error in SerpAPI reverse image search: {e}")
        # Fall back to caption-based search
        return await _google_search_by_keyword()

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
async def google_search(query: str or List[str], max_results=5) -> List[Dict[str, Any]]:
    """
    Enhanced Google search with caching and error handling.
    
    Args:
        query: Search query or list of queries to run in parallel
        max_results: Maximum number of results to return per query
        
    Returns:
        List of product dictionaries
    """
    if not SERP_KEY:
        print("No SerpAPI key provided, skipping Google search")
        return []
    
    # Handle both string and list inputs
    if isinstance(query, str):
        queries = [query]
    else:
        queries = query
        
    print(f"Running Google Shopping search for: {queries}")
    
    if is_enabled("PARALLEL_CAPTION_SEARCH") and len(queries) > 1:
        # Run searches in parallel
        tasks = [_run_single_google_search(q, max_results) for q in queries]
        results_lists = await asyncio.gather(*tasks)
        
        # Combine results, removing duplicates
        combined_results = []
        seen_urls = set()
        
        for results in results_lists:
            for item in results:
                if "url" in item and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    combined_results.append(item)
        
        # Sort by relevance (based on result order) and return
        return combined_results[:max_results]
    else:
        # Run just the first query
        return await _run_single_google_search(queries[0], max_results)

async def _run_single_google_search(query: str, max_results=5) -> List[Dict[str, Any]]:
    """Helper function to run a single Google Shopping search"""
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
async def amazon_search(query: str or List[str], max_hits=20, page: int = 1, country: str = "US", sort_by: str = "RELEVANCE", product_condition: str = "ALL", is_prime: bool = False, deals_and_discounts: str = "NONE", **kwargs) -> list[dict]:
    """
    Search Amazon for products based on a text query with RapidAPI parameters
    
    Args:
        query: Search query or list of queries to run in parallel
        max_hits: Maximum number of results to return
        page: Page number for pagination
        country: Country code for Amazon site
        sort_by: Sort method (RELEVANCE, PRICE_LOW_TO_HIGH, etc.)
        product_condition: Filter by product condition (ALL, NEW, USED, etc.)
        is_prime: Filter for Prime-eligible items
        deals_and_discounts: Filter for deals (NONE, DEALS, TODAY_DEALS)
        **kwargs: Additional filters (category_id, min_price, etc.)
        
    Returns:
        List of product dictionaries
    """
    if not AMAZON_RAPID_KEY or AMAZON_RAPID_KEY == "732af5ccd8msh9d3c01da8429b19p1f38d8jsnc3b4b20a28d8":
        print("No valid Amazon Rapid API key provided, skipping Amazon search")
        return []
        
    # Handle both string and list inputs
    if isinstance(query, str):
        queries = [query]
    else:
        queries = query
        
    print(f"Searching Amazon for: {queries} (page={page}, country={country})")
    
    if is_enabled("PARALLEL_CAPTION_SEARCH") and len(queries) > 1:
        # Run searches in parallel
        tasks = [_run_single_amazon_search(q, max_hits, page, country, sort_by, 
                                         product_condition, is_prime, 
                                         deals_and_discounts, **kwargs) 
                for q in queries]
        results_lists = await asyncio.gather(*tasks)
        
        # Combine results, removing duplicates
        combined_results = []
        seen_urls = set()
        
        for results in results_lists:
            for item in results:
                if "url" in item and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    combined_results.append(item)
        
        print(f"Combined {len(combined_results)} Amazon results from {len(queries)} queries")
        return combined_results[:max_hits]
    else:
        # Run just the first query
        return await _run_single_amazon_search(queries[0], max_hits, page, country, 
                                            sort_by, product_condition, is_prime, 
                                            deals_and_discounts, **kwargs)

async def _run_single_amazon_search(query: str, max_hits=20, page: int = 1, country: str = "US", 
                                  sort_by: str = "RELEVANCE", product_condition: str = "ALL", 
                                  is_prime: bool = False, deals_and_discounts: str = "NONE", 
                                  **kwargs) -> list[dict]:
    """Helper function to run a single Amazon search"""
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

async def simple_caption(img_path: str, object_label: str = None) -> List[str]:
    """
    Generate multiple rich captions for an image using available models and techniques.
    
    Args:
        img_path: Path to the image
        object_label: Optional class label from object detection
    
    Returns:
        List of caption strings for search, from most to least specific
    """
    captions = []
    
    # If we have the object label, use it for a basic caption
    if object_label and object_label.lower() not in ("unknown", "custom"):
        basic_caption = f"modern {object_label} for living room"
        captions.append(basic_caption)
    
    # Check if GPT-4o caption generation is enabled
    if is_enabled("GPT4O_CAPTIONS"):
        try:
            # Try to use OpenAI's gpt-4o if key is available
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
                
                # First get detailed description
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a furniture and interior design expert."},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Describe this furniture item in 5-7 words. Focus on style, color, material, and specific type."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                            ]}
                        ],
                        max_tokens=30
                    )
                    
                    gpt_caption = response.choices[0].message.content.strip()
                    captions.append(gpt_caption)
                    
                    # Also get material info
                    if is_enabled("MATERIAL_DETECTOR"):
                        material_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a furniture and interior design expert."},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "What material is this furniture made of? Answer with just one word."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                                ]}
                            ],
                            max_tokens=10
                        )
                        
                        material = material_response.choices[0].message.content.strip().lower()
                        if material and len(material) < 20:  # Sanity check on response
                            if object_label:
                                material_caption = f"{material} {object_label}"
                                captions.append(material_caption)
                    
                    print(f"Generated GPT-4o caption: {gpt_caption}")
                except Exception as e:
                    print(f"Error with detailed GPT caption: {e}")
                    
        except Exception as e:
            print(f"Error with GPT-4o caption generation: {e}")
    
    # Try color extraction if enabled
    if is_enabled("COLOR_EXTRACTOR"):
        try:
            from utils.image_color import get_material_color_keywords
            
            color_info = get_material_color_keywords(img_path)
            color = color_info.get("color", "")
            material = color_info.get("material", "")
            
            if color and object_label:
                color_caption = f"{color} {object_label}"
                captions.append(color_caption)
                
            if material and object_label and material != "unknown":
                material_caption = f"{material} {object_label}"
                if material_caption not in captions:
                    captions.append(material_caption)
                    
            print(f"Extracted color keywords: {color_info}")
        except Exception as e:
            print(f"Error with color extraction: {e}")
            
    # Fallback: Use simple heuristics based on the image dimensions
    if not captions:
        try:
            img = Image.open(img_path)
            width, height = img.size
            aspect_ratio = width / height
            
            if aspect_ratio > 1.5:
                captions.append("wide rectangular furniture item")
            elif aspect_ratio < 0.7:
                captions.append("tall furniture piece")
            elif width > 300 and height > 300:
                captions.append("large furniture item home decor")
            else:
                captions.append("small furniture decor item")
        except Exception as e:
            print(f"Error generating caption based on dimensions: {e}")
            captions.append("furniture home decor item")
    
    # If we still have no captions, add a default one
    if not captions:
        captions.append("furniture home decor item")
        
    # Remove duplicates while preserving order
    unique_captions = []
    for caption in captions:
        if caption not in unique_captions:
            unique_captions.append(caption)
            
    print(f"Generated captions: {unique_captions}")
    return unique_captions

@retry_async(max_retries=3)
async def verify_product_url(url: str, original_image_path: str = None) -> Dict[str, Any]:
    """
    Verify a product URL and extract schema data with more lenient validation.
    If an original image is provided and SSIM validation is enabled, will compare
    product images for visual similarity.
    
    Args:
        url: URL to verify
        original_image_path: Path to the original product image for comparison
        
    Returns:
        Dictionary with product information and validation status
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
            "valid": True,
            "confidence": 0.8  # Reasonable default confidence
        }
    
    # For URLs from Google Shopping results, trust them more
    if url.startswith("https://www.google.") and "/shopping/" in url:
        return {
            "url": url,
            "title": "Shopping Product",
            "price": "See website for price",
            "currency": "USD",
            "retailer": "Google Shopping",
            "valid": True,
            "confidence": 0.7  # Slightly lower confidence than direct product URLs
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
                "valid": True,
                "confidence": 0.75  # Good default confidence for known retailers
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
            
            # Initialize variables
            product_data = {
                "url": url,
                "title": "Unknown Product",
                "price": "See website for price",
                "currency": "USD",
                "retailer": urlparse(url).netloc.split(":")[0].replace("www.", ""),
                "valid": True,
                "confidence": 0.6  # Default confidence
            }
            
            # Try to extract product schema data
            try:
                data = extruct.extract(r.text, base_url=url)
                prod = next((x for x in data["json-ld"] if x.get("@type") == "Product"), {})
                
                if prod:
                    offers = prod.get("offers", {})
                    product_data.update({
                        "title": prod.get("name", product_data["title"]),
                        "price": offers.get("price", product_data["price"]),
                        "currency": offers.get("priceCurrency", product_data["currency"]),
                        "confidence": 0.8  # Higher confidence for schema-verified products
                    })
                    print(f"  Valid product: {product_data['title']} ({product_data['price']} {product_data['currency']})")
            except Exception as schema_error:
                print(f"  Error extracting schema: {schema_error}")
            
            # If SSIM validation is enabled and we have an original image, compare product images
            if is_enabled("SSIM_PRODUCT_VALIDATION") and original_image_path:
                try:
                    from bs4 import BeautifulSoup
                    from skimage.metrics import structural_similarity as ssim
                    import cv2
                    
                    # Parse the HTML
                    soup = BeautifulSoup(r.text, 'html.parser')
                    
                    # Try to find product images
                    product_image_url = None
                    
                    # First try OpenGraph image
                    og_image = soup.find('meta', property='og:image')
                    if og_image and og_image.get('content'):
                        product_image_url = og_image.get('content')
                    
                    # If not found, try other common image elements
                    if not product_image_url:
                        # Look for product image in common containers
                        img_tags = []
                        for selector in ['#product-image', '.product-image', '.product-img', '#main-image', '.main-image']:
                            img_tag = soup.select_one(selector)
                            if img_tag:
                                img_tags.append(img_tag)
                                
                        # Also try to find largest image on the page
                        all_imgs = soup.find_all('img')
                        largest_img = None
                        largest_area = 0
                        
                        for img in all_imgs:
                            # Check for common product image attributes
                            if any(attr in img.get('class', []) for attr in ['product', 'main', 'hero', 'featured']):
                                img_tags.append(img)
                                
                            # Check for large images
                            width = img.get('width')
                            height = img.get('height')
                            if width and height:
                                try:
                                    w, h = int(width), int(height)
                                    area = w * h
                                    if area > largest_area:
                                        largest_area = area
                                        largest_img = img
                                except ValueError:
                                    pass
                        
                        if largest_img:
                            img_tags.append(largest_img)
                            
                        # Extract URLs from found images
                        for img in img_tags:
                            if img.get('src'):
                                product_image_url = img.get('src')
                                # Make absolute URL if relative
                                if not product_image_url.startswith(('http://', 'https://')):
                                    base_url = get_base_url(url, r.text)
                                    product_image_url = urljoin(base_url, product_image_url)
                                break
                    
                    # If we found a product image, download and compare
                    if product_image_url:
                        print(f"  Found product image: {product_image_url}")
                        try:
                            # Download product image
                            async with httpx.AsyncClient(timeout=10) as img_client:
                                img_response = await img_client.get(product_image_url)
                                if img_response.status_code == 200:
                                    # Save product image temporarily
                                    product_img_path = "temp_product_image.jpg"
                                    with open(product_img_path, "wb") as f:
                                        f.write(img_response.content)
                                    
                                    # Load images for comparison
                                    original_img = cv2.imread(original_image_path)
                                    product_img = cv2.imread(product_img_path)
                                    
                                    if original_img is not None and product_img is not None:
                                        # Resize to match
                                        h, w = original_img.shape[:2]
                                        product_img = cv2.resize(product_img, (w, h))
                                        
                                        # Convert to grayscale for SSIM
                                        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                                        product_gray = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)
                                        
                                        # Calculate SSIM
                                        similarity_score, _ = ssim(original_gray, product_gray, full=True)
                                        print(f"  SSIM score: {similarity_score:.2f}")
                                        
                                        # Update confidence based on similarity
                                        if similarity_score > 0.4:  # Reasonable threshold for furniture
                                            product_data["confidence"] = max(product_data["confidence"], similarity_score)
                                            print(f"  Matched with confidence: {similarity_score:.2f}")
                                            
                                            # Add image to product data
                                            product_data["image"] = product_image_url
                        except Exception as img_error:
                            print(f"  Error comparing images: {img_error}")
                except Exception as bs_error:
                    print(f"  Error extracting product images: {bs_error}")
            
            return product_data
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
        
        # Generate captions for the image using the object class if available
        captions = await simple_caption(crop_path, obj_class)
        
        # Use additional metadata if provided
        if object_metadata:
            print(f"Using additional object metadata: {object_metadata}")
            # We could use the bbox/dimensions for more targeted search
            if 'class' in object_metadata and not obj_class:
                obj_class = object_metadata['class']
                # Regenerate captions with the class
                captions = await simple_caption(crop_path, obj_class)
        
        # Print the captions for debugging
        caption_str = ", ".join(captions) if isinstance(captions, list) else captions
        print(f"Caption(s): {caption_str}")
        
        # Start concurrent API calls
        print("Starting concurrent API searches...")
        reverse_image_task = google_reverse_image(img_bytes)
        amazon_search_task = amazon_search(captions, max_results)
        google_search_task = google_search(captions, max_results)
        
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
                    "image": item.get("thumbnail", item.get("image", "")),
                    "confidence": 0.8  # Good default confidence for Google Shopping results
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
                    "valid": True,
                    "confidence": 0.75  # Good default confidence for Amazon results
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
                
                verification_tasks = [verify_product_url(url, crop_path) for url in batch_urls]
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
            product_type = obj_class if obj_class else captions[0]
            
            fallback_products = [
                {
                    "url": f"https://www.amazon.com/s?k={product_type.replace(' ', '+')}+furniture",
                    "title": f"{product_type.title()} on Amazon",
                    "price": "Various prices",
                    "currency": "USD",
                    "retailer": "Amazon",
                    "source": "fallback",
                    "valid": True,
                    "confidence": 0.3  # Low confidence for fallback results
                },
                {
                    "url": f"https://www.wayfair.com/keyword.php?keyword={product_type.replace(' ', '+')}",
                    "title": f"{product_type.title()} on Wayfair",
                    "price": "Various prices",
                    "currency": "USD",
                    "retailer": "Wayfair",
                    "source": "fallback",
                    "valid": True,
                    "confidence": 0.3
                },
                {
                    "url": f"https://www.ikea.com/us/en/search/?q={product_type.replace(' ', '+')}",
                    "title": f"{product_type.title()} on IKEA",
                    "price": "Various prices",
                    "currency": "USD",
                    "retailer": "IKEA",
                    "source": "fallback",
                    "valid": True,
                    "confidence": 0.3
                }
            ]
            
            final_products.extend(fallback_products)
        
        # Apply enhanced ranking logic if enabled
        if is_enabled("NEW_PRODUCT_RANKER"):
            # Check for color information from the original caption
            color_terms = []
            material_terms = []
            
            # Extract color and material terms from captions
            for cap in captions if isinstance(captions, list) else [captions]:
                cap_lower = cap.lower()
                # Extract potential color terms
                for color in ["black", "white", "gray", "grey", "brown", "blue", "red", 
                             "green", "yellow", "purple", "pink", "orange", "beige", 
                             "walnut", "oak", "mahogany", "cherry", "maple", "teak"]:
                    if color in cap_lower:
                        if color in ["walnut", "oak", "mahogany", "cherry", "maple", "teak"]:
                            material_terms.append(color)
                        else:
                            color_terms.append(color)
            
            # Try to get additional metadata from the object_metadata
            if object_metadata:
                class_name = object_metadata.get('class', '')
                confidence = object_metadata.get('confidence', 0.0)
                
                # If the object was detected with high confidence, boost its products
                if confidence > 0.7:
                    for product in final_products:
                        if "confidence" not in product:
                            product["confidence"] = 0.6 + (confidence - 0.7)
            
            try:
                # If we have color extraction data available, use it
                if is_enabled("COLOR_EXTRACTOR") and Path(crop_path).exists():
                    from utils.image_color import get_material_color_keywords
                    
                    # Extract color and material information
                    color_info = get_material_color_keywords(crop_path)
                    if color_info:
                        if "color" in color_info:
                            # Add specific color terms from extraction
                            color_terms.extend(color_info["color"].split("-"))
                        
                        if "material" in color_info and color_info["material"] != "unknown":
                            material_terms.append(color_info["material"])
            except Exception as e:
                print(f"Error extracting color/material for ranking: {e}")
                
            print(f"Ranking with color terms: {color_terms} and material terms: {material_terms}")
                
            # Sort products based on the new ranking logic
            final_products.sort(key=lambda x: (
                # Primary ranking criteria
                0 if x.get("source") != "fallback" else 1,  # Real products first
                
                # Secondary criteria - confidence score (higher is better)
                -float(x.get("confidence", 0.5)),  # Default 0.5 if not set
                
                # Check if product has price (prefer products with specific prices)
                0 if x.get("price") and x.get("price") != "See website for price" 
                  and x.get("price") != "Various prices" else 1,
                
                # Check color match in title or description
                not any(term in x.get("title", "").lower() for term in color_terms) 
                  and not any(term in x.get("description", "").lower() for term in color_terms),
                
                # Check material match in title or description  
                not any(term in x.get("title", "").lower() for term in material_terms)
                  and not any(term in x.get("description", "").lower() for term in material_terms),
                
                # Final ranking criteria
                0 if x.get("image") else 1,  # Products with image before those without
            ))
        else:
            # Original ranking logic
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
                "valid": True,
                "confidence": 0.1  # Very low confidence for error fallbacks
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