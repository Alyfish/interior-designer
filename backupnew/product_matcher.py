import os
import json
import logging
import re
from typing import Optional, Dict, List, Any
import numpy as np

# LangChain components
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import PromptTemplate
# GoogleSearch will be lazily imported inside the tool to avoid hard import errors if the package is missing

# +++ BEGIN ADDED CODE FOR DEBUGGING KEY +++
from pathlib import Path
from dotenv import load_dotenv

# Determine the directory of this script new_product_matcher.py
_NPM_SCRIPT_DIR = Path(__file__).resolve().parent
# Path to the .env file in the parent directory (C:/Users/aly17/new/.env)
_NPM_DOTENV_PATH_PROJECT_ROOT = _NPM_SCRIPT_DIR.parent / ".env"

_npm_loaded_from_project_root = False
if _NPM_DOTENV_PATH_PROJECT_ROOT.exists():
    if load_dotenv(_NPM_DOTENV_PATH_PROJECT_ROOT, encoding='utf-16', override=True):
        logging.info(f"NPM: Successfully re-loaded .env from: {_NPM_DOTENV_PATH_PROJECT_ROOT}")
        _npm_loaded_from_project_root = True
    else:
        logging.warning(f"NPM: Found .env at {_NPM_DOTENV_PATH_PROJECT_ROOT}, but failed to re-load.")
else:
    logging.warning(f"NPM: .env not found at {_NPM_DOTENV_PATH_PROJECT_ROOT} for re-load attempt.")

if not _npm_loaded_from_project_root:
    _NPM_DOTENV_PATH_SCRIPT_DIR = _NPM_SCRIPT_DIR / ".env"
    if _NPM_DOTENV_PATH_SCRIPT_DIR.exists():
        if load_dotenv(_NPM_DOTENV_PATH_SCRIPT_DIR, encoding='utf-16', override=True):
            logging.info(f"NPM: Successfully re-loaded .env from: {_NPM_DOTENV_PATH_SCRIPT_DIR}")
        else:
            logging.warning(f"NPM: Found .env at {_NPM_DOTENV_PATH_SCRIPT_DIR}, but failed to re-load.")
    else:
        logging.warning(f"NPM: .env not found at {_NPM_DOTENV_PATH_SCRIPT_DIR} for re-load attempt.")

# Re-fetch SERP_API_KEY directly after re-load attempt
SERP_API_KEY_NPM = os.getenv("SERP_API_KEY", "")
logging.info(f"NPM: SERP_API_KEY after explicit re-load/os.getenv: {'Yes' if SERP_API_KEY_NPM else 'No'}. Starts with: {SERP_API_KEY_NPM[:5]}...")
# +++ END ADDED CODE FOR DEBUGGING KEY +++

# Configuration for API keys
try:
    from config import OPENAI_API_KEY, SERP_API_KEY, SEARCH_API_KEY, IMGBB_API_KEY, check_api_keys
    logging.info(f"NPM: SERP_API_KEY imported from config.py starts with: {SERP_API_KEY[:5]}...")
except ImportError:
    logging.error("NPM: Failed to import from config.py")
    # Fallback if config import fails, though SERP_API_KEY_NPM should be available
    SERP_API_KEY = SERP_API_KEY_NPM # Ensure it's defined
    SEARCH_API_KEY = SERP_API_KEY_NPM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") # Attempt to get other keys too
    IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "")
    def check_api_keys(): pass # Dummy function

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if essential API keys are loaded upon module import
check_api_keys()

# --- Feature Flags --- #
ENABLE_REVERSE_IMAGE_SEARCH = os.getenv("REVERSE_IMAGE_SEARCH", "off").lower() in ["on", "true", "1", "yes"]

# --- Constants --- #
# Updated to reflect a more general product search capability
AGENT_PROMPT_TEMPLATE = """
You are a furniture search specialist helping customers find matching products online. 

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT INSTRUCTIONS FOR FURNITURE SEARCH:
1. When given furniture descriptions, analyze the color, style, size, shape, material, and key features
2. Create detailed search queries that include multiple descriptive keywords like:
   - Colors: "black leather", "white oak", "grey fabric", "walnut wood"
   - Styles: "mid-century modern", "contemporary", "rustic farmhouse", "scandinavian"
   - Materials: "leather", "fabric upholstered", "solid wood", "metal frame"
   - Features: "accent chair", "sectional sofa", "extendable dining table", "storage ottoman"
3. Use search terms that furniture stores commonly use in product titles
4. Combine multiple attributes in your search: "mid-century modern walnut dining table" instead of just "table"

When providing the final answer, list products as a numbered list with title, price, store, and link if available. 
Format each product like: '1. Product Name - Price: $XX.XX - Store: Store Name - Link: https://...'.
If you cannot find specific information like price or store, state 'Price not available' or 'Store not specified'.
If no products are found, state 'No products found matching your query.'

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# --- Tool Definition --- #
def search_products_serpapi_tool(query: str, serp_api_key: str) -> str:
    """
    Use SerpAPI to search for products on Google Shopping.
    Returns formatted product results with real URLs.
    """
    logger.info(f"ProductSearchTool: Searching with query: '{query}'")
    
    # If no API key is provided, return mock results for testing
    if not serp_api_key or serp_api_key == "your_serpapi_key_here":
        logger.warning("No valid SERP API key provided. Returning mock results.")
        return generate_mock_products(query)
    
    logger.info(f"ProductSearchTool: Attempting to use SERP_API_KEY starting with: {str(serp_api_key)[:5]}...")
    try:
        # Try the correct import for google-search-results package
        GoogleSearch = None
        try:
            from serpapi.google_search import GoogleSearch
            logger.info("Successfully imported GoogleSearch from serpapi.google_search")
        except ImportError:
            try:
                from serpapi import GoogleSearch
                logger.info("Successfully imported GoogleSearch from serpapi")
            except ImportError:
                try:
                    from google_search_results import GoogleSearch
                    logger.info("Successfully imported GoogleSearch from google_search_results")
                except ImportError:
                    logger.error("Could not import GoogleSearch from any known package")
                    return generate_mock_products(query)

        search = GoogleSearch({
            "engine": "google_shopping",
            "q": query,
            "api_key": serp_api_key,
            "num": 10
        })
        results = search.get_dict()
        
        logger.info(f"SerpAPI returned: {results.keys()}")
        
        # If the response contains an explicit error from SerpAPI, surface it for easier debugging
        if "error" in results:
            err_msg = results.get("error", "Unknown SerpAPI error")
            logger.error(f"SerpAPI ERROR (Text Search): {err_msg}")
            return f"SerpAPI error (Text Search): {err_msg}"
        
        if "shopping_results" not in results:
            logger.warning(f"No shopping_results in response. Available keys: {list(results.keys())}")
            return f"No shopping results found for '{query}'"
        
        products = []
        shopping_results = results["shopping_results"]
        logger.info(f"Found {len(shopping_results)} shopping results")
        
        for i, item in enumerate(shopping_results[:5], 1):
            title = item.get("title", "Unknown Product")
            price = item.get("price", "Price not available")
            source = item.get("source", "Unknown Store")
            
            # Try multiple URL fields that SerpAPI might use
            link = (item.get("link") or 
                   item.get("product_link") or 
                   item.get("url") or
                   item.get("product_url") or
                   f"https://www.google.com/search?tbm=shop&q={query.replace(' ', '+')}")
            
            # If link is still "#" or empty, create a Google Shopping search URL
            if not link or link == "#" or link == "":
                link = f"https://www.google.com/search?tbm=shop&q={title.replace(' ', '+')}"
            
            logger.info(f"Product {i}: title='{title}', price='{price}', source='{source}', link='{link}'")
            
            products.append(f"{i}. {title} - Price: {price} - Store: {source} - Link: {link}")
        
        if not products:
            return f"No products found for query: {query}"
        
        result = "\n".join(products)
        logger.info(f"Returning formatted products: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in product search: {e}", exc_info=True)
        return generate_mock_products(query)

def generate_mock_products(query: str) -> str:
    """
    Generate mock product results for testing when SERP API is not available.
    """
    logger.info(f"Generating mock products for query: '{query}'")
    
    mock_products = [
        f"1. {query.title()} - Modern Style - Price: $299.99 - Store: Wayfair - Link: https://www.wayfair.com/search?query={query.replace(' ', '%20')}",
        f"2. {query.title()} - Contemporary Design - Price: $449.99 - Store: IKEA - Link: https://www.ikea.com/us/en/search/products/?q={query.replace(' ', '%20')}",
        f"3. {query.title()} - Premium Quality - Price: $699.99 - Store: West Elm - Link: https://www.westelm.com/search/results.html?words={query.replace(' ', '%20')}",
        f"4. {query.title()} - Budget Friendly - Price: $149.99 - Store: Amazon - Link: https://www.amazon.com/s?k={query.replace(' ', '+')}",
        f"5. {query.title()} - Designer Collection - Price: $899.99 - Store: CB2 - Link: https://www.cb2.com/search?query={query.replace(' ', '%20')}"
    ]
    
    result = "\n".join(mock_products)
    logger.info(f"Generated mock products: {result}")
    return result

# --- NEW: Reverse Image Search Functions --- #

def search_products_reverse_image_serpapi(image_path: str, serp_api_key: str, query_text: str = "") -> str:
    """
    Use SerpAPI Google Reverse Image for reverse image search to find similar products.
    Based on: https://serpapi.com/google-reverse-image-api
    
    Args:
        image_path: Path to the furniture crop image
        serp_api_key: SerpAPI key for Google Reverse Image search
        query_text: Optional text query to enhance search
    
    Returns:
        Formatted string of product results
    """
    logger.info(f"Reverse image search for: {image_path} using SerpAPI Google Reverse Image.")
    
    if not serp_api_key or serp_api_key == "your_serpapi_key_here":
        logger.warning("No valid SerpAPI key for reverse image search. Returning mock results.")
        return generate_mock_products(f"visual search {query_text}")
    
    logger.info(f"ReverseImageSearch: Attempting to use SERP_API_KEY starting with: {str(serp_api_key)[:5]}...")
    try:
        # Upload image and get URL (using existing function)
        image_url = _upload_image_to_public_url(image_path)
        if not image_url:
            logger.warning("Failed to upload image for reverse search")
            return generate_mock_products(f"visual search {query_text}")

        # Try the correct import for google-search-results package
        GoogleSearch = None
        try:
            from serpapi.google_search import GoogleSearch
            logger.info("Successfully imported GoogleSearch from serpapi.google_search")
        except ImportError:
            try:
                from serpapi import GoogleSearch
                logger.info("Successfully imported GoogleSearch from serpapi")
            except ImportError:
                try:
                    from google_search_results import GoogleSearch
                    logger.info("Successfully imported GoogleSearch from google_search_results")
                except ImportError:
                    logger.error("Could not import GoogleSearch from any known package")
                    return generate_mock_products(f"visual search {query_text}")

        # SerpAPI Google Lens parameters (preferred over legacy reverse image)
        # Using Google Lens usually provides a "visual_matches" array that contains
        # product-oriented results (title, link, price, source, thumbnail, …).
        search_params = {
            "engine": "google_lens",          # switch to Lens engine for richer data
            "url": image_url,                  # public image URL obtained from ImgBB
            "search_type": "products",       # ask specifically for product matches
            "api_key": serp_api_key,
            "gl": "us",
            "hl": "en"
        }
        # Add an auxiliary text hint (helps Lens rank results) if provided
        if query_text.strip():
            search_params["q"] = query_text
            
        logger.info(f"SerpAPI Google Lens request params: {search_params}")
        
        # Make request to SerpAPI
        search = GoogleSearch(search_params)
        results = search.get_dict()
        
        logger.info(f"SerpAPI returned keys: {list(results.keys())}")
        
        # If the response contains an explicit error from SerpAPI, surface it for easier debugging
        if "error" in results:
            err_msg = results.get("error", "Unknown SerpAPI error")
            logger.error(f"SerpAPI ERROR: {err_msg}")
            return f"SerpAPI error: {err_msg}"
        
        # Parse visual matches from SerpAPI response
        products = []
        visual_matches = results.get("visual_matches", [])
        # Fall back to legacy key if Lens doesn't return visual_matches
        if not visual_matches:
            visual_matches = results.get("image_results", [])  # backward compatibility
        
        if not visual_matches:
            logger.warning("No visual matches found in SerpAPI response")
            return "No visual matches found for the uploaded furniture image"
        
        logger.info(f"Found {len(visual_matches)} visual matches")
        
        for i, item in enumerate(visual_matches[:5], 1):
            title = item.get("title", "Unknown Product")
            
            # Extract domain/source from link
            link = item.get("link", "#")
            # Google Lens may use either "source" or "displayed_link" depending on the match type
            source = item.get("source") or item.get("displayed_link") or "Unknown Store"
            if not source or source == "Unknown Store":
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(link)
                    source = parsed.netloc.replace('www.', '')
                except:
                    source = "Unknown Store"
            
            # Price is not typically available in reverse image results
            price = "Price not available"
            
            logger.info(f"Visual match {i}: title='{title}', source='{source}', link='{link}'")
            
            products.append(f"{i}. {title} - Price: {price} - Store: {source} - Link: {link}")
        
        if not products:
            return f"No products found in visual search results"
        
        result = "\n".join(products)
        logger.info(f"Returning visual search products: {len(products)} items")
        return result

    except Exception as e:
        logger.error(f"Error in reverse image search: {e}", exc_info=True)
        return generate_mock_products(f"visual search {query_text}")

def search_products_hybrid(image_path: str, caption_data: dict, serp_api_key_text: str, serp_api_key_visual: str = None) -> str:
    """
    Combines results from visual search (SerpAPI Google Reverse Image) and text-based search (SerpAPI Google Shopping).

    Args:
        image_path: Path to the furniture crop image.
        caption_data: Dictionary containing style, material, etc. from image analysis.
        serp_api_key_text: API key for SerpAPI (text search).
        serp_api_key_visual: API key for SerpAPI (visual search) - can be same as text key.

    Returns:
        Combined formatted string of product results.
    """
    logger.info("Starting hybrid search (visual + text)")
    visual_results_str = ""
    text_results_str = ""

    # 1. Visual Search using same SERP API key
    # Use the same key for visual search if no separate key provided
    visual_search_key = serp_api_key_visual or serp_api_key_text
    
    if image_path and visual_search_key:
        # Construct a query text from caption data for better visual search results
        
        # --- NEW Improved Visual Query Construction ---
        detailed_caption = caption_data.get('caption', '').strip()
        style = caption_data.get('style', '').strip()
        
        # Start with the most descriptive text available: the GPT-4V caption.
        # Then, add the style if it's not already mentioned, as style is a very strong keyword.
        visual_query_text = detailed_caption
        if style and style.lower() != 'unknown' and style.lower() not in detailed_caption.lower():
            visual_query_text = f"{style} {detailed_caption}"
        
        # Clean up common unhelpful words and extra spaces
        visual_query_text = visual_query_text.replace("unknown", "").replace("  ", " ").strip()
        
        # Fallback if the query is empty after cleaning
        if not visual_query_text:
            visual_query_text = "furniture"
            
        logger.info(f"NPM: Constructed visual_query_text: '{visual_query_text}'")
        # --- END Improved Visual Query Construction ---
        
        visual_results_str = search_products_reverse_image_serpapi(
            image_path=image_path, 
            serp_api_key=visual_search_key, 
            query_text=visual_query_text
        )
    else:
        logger.info("Skipping visual search due to missing image path or API key.")

    # 2. Text-based Search using SerpAPI (existing logic)
    # --- NEW Improved Text Query Construction ---
    detailed_caption_text = caption_data.get('caption', '').strip()
    style_text = caption_data.get('style', '').strip()

    # Start with the detailed caption, and prepend the style if not already present.
    text_search_query = detailed_caption_text
    if style_text and style_text.lower() != 'unknown' and style_text.lower() not in detailed_caption_text.lower():
        text_search_query = f"{style_text} {text_search_query}"
        
    # Clean up and provide a fallback
    text_search_query = text_search_query.replace("unknown", "").replace("  ", " ").strip()
    if not text_search_query:
        text_search_query = "modern furniture" # Default fallback
    logger.info(f"NPM: Constructed text_search_query: '{text_search_query}'")
    # --- END Improved Text Query Construction ---

    text_results_str = search_products_serpapi_tool(query=text_search_query, serp_api_key=serp_api_key_text)

    # 3. Combine Results
    combined_results = []
    if visual_results_str and "No visual matches found" not in visual_results_str and "mock products" not in visual_results_str:
        combined_results.append("=== VISUAL SEARCH RESULTS ===")
        combined_results.append(visual_results_str)
    
    if text_results_str and "No products found" not in text_results_str and "mock products" not in text_results_str:
        if combined_results: # Add separator if visual results are present
            combined_results.append("\n=== TEXT SEARCH RESULTS (based on image features) ===")
        else:
            combined_results.append("=== TEXT SEARCH RESULTS (based on image features) ===")
        combined_results.append(text_results_str)
    
    if not combined_results:
        return "No products found from either visual or text search."
        
    return "\n".join(combined_results)

# --- Agent Creation --- #
def create_new_product_search_agent():
    """
    Creates and returns a new LangChain agent for product searching.
    The agent will have access to the standard product search tool.
    If reverse image search is enabled and configured, it will also have the visual search tool.
    """
    logger.info("Creating new product search agent...")
    # The agent needs access to the API keys. We pass them when the tool is used.
    # Note: Direct use of API keys in tool definition lambda can be risky if not handled carefully.
    # Here, we ensure keys are loaded from config and passed at call time.
    
    # Ensure keys are loaded from config
    openai_key = OPENAI_API_KEY
    serp_key_text_search = SERP_API_KEY
    searchapi_key_visual_search = SEARCH_API_KEY # Updated name
    imgbb_key = IMGBB_API_KEY

    if not openai_key:
        logger.error("OpenAI API key not found. Agent creation failed.")
        return None

    # Standard text-based product search tool
    tools = [
        Tool(
            name="ProductSearch",
            func=lambda query: search_products_serpapi_tool(query, serp_api_key=serp_key_text_search),
            description="Searches for products based on a text description (e.g., 'red leather sofa'). Provides product name, price, store, and link.",
        )
    ]

    # Visual Product Search Tool (conditionally added)
    if ENABLE_REVERSE_IMAGE_SEARCH:
        logger.info(f"Reverse image search is ENABLED by feature flag.")
        if serp_key_text_search and imgbb_key:  # Using same SerpAPI key for both
            logger.info("SerpAPI key and ImgBB key are available. Adding VisualProductSearch tool.")
            tools.append(
                Tool(
                    name="VisualProductSearch",
                    # This tool expects an image_path. The agent will need to be prompted for this,
                    # or it needs to be passed in the initial agent invocation if a primary image is being searched.
                    # For now, let's assume image_path is provided in the query or context somehow.
                    # The input to this tool should ideally be the image_path and optional query_text.
                    # The agent prompt might need adjustment if the agent is to decide to use this tool.
                    func=lambda input_str: search_products_reverse_image_serpapi(image_path=input_str, serp_api_key=serp_key_text_search),
                    description=(
                        "Performs a reverse image search for a product given a local image path using SerpAPI Google Reverse Image. "
                        "Use this tool when you have an image of the product you want to find. "
                        "Input should be the direct file path to the image (e.g., 'path/to/image.jpg')."
                    )
                )
            )
        else:
            missing_for_visual = []
            if not serp_key_text_search: missing_for_visual.append("SerpAPI Key")
            if not imgbb_key: missing_for_visual.append("ImgBB Key")
            logger.warning(f"Reverse image search enabled, but VisualProductSearch tool cannot be added. Missing: {', '.join(missing_for_visual)}. Will rely on text search only.")
    else:
        logger.info("Reverse image search is DISABLED by feature flag.")

    if not tools:
        logger.error("No tools available for the agent. Agent creation aborted.")
        return None

    try:
        prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
        agent = create_react_agent(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_key), tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        logger.info(f"✅ Product Search LangChain agent created successfully with {len(tools)} tools.")
        return agent_executor
    except Exception as e:
        logger.error(f"❌ Failed to create Product Search agent: {e}", exc_info=True)
        return None

# --- Visual Search Tool (NEW) ---
def _upload_image_to_public_url(image_path: str) -> Optional[str]:
    """
    Upload an image to ImgBB (free service) to get a publicly accessible URL.
    
    Args:
        image_path: Local path to the image file
        
    Returns:
        Public URL of the uploaded image, or None if upload fails
    """
    import requests
    import base64
    
    # ImgBB API key - this should be in config.py or environment variables
    # For now, using a placeholder. In production, add IMGBB_API_KEY to config
    IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "")
    
    if not IMGBB_API_KEY:
        logger.warning("IMGBB_API_KEY not set. Cannot upload image for reverse search.")
        # For testing purposes, return a placeholder URL if the file exists
        if os.path.exists(image_path):
            logger.info("Image exists locally, but cannot upload. Using placeholder for testing.")
            # Return None to indicate we can't proceed with actual reverse search
            return None
        return None
    
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None
            
        # Read and encode image to base64
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # ImgBB API endpoint
        url = "https://api.imgbb.com/1/upload"
        
        payload = {
            'key': IMGBB_API_KEY,
            'image': image_data,
            'expiration': 3600  # 1 hour expiration for privacy
        }
        
        logger.info(f"Uploading image {image_path} to ImgBB...")
        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('success') and result.get('data', {}).get('url'):
            public_url = result['data']['url']
            logger.info(f"✅ Image uploaded successfully: {public_url}")
            return public_url
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            logger.error(f"ImgBB upload failed: {error_msg}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error uploading to ImgBB: {e}")
        return None
    except Exception as e:
        logger.error(f"Error uploading image to ImgBB: {e}", exc_info=True)
        return None

def search_products_visual_serpapi(image_path: str, query_text: str, serp_api_key: str) -> list:
    """
    Use SerpAPI Google Lens to search for products based on an image and text query.
    Returns a list of product dictionaries.
    """
    logger.info(f"VisualProductSearch: Searching with image: '{image_path}' and text: '{query_text}'")
    products = []

    # 1. Upload image to get a public URL (placeholder)
    # public_image_url = _upload_image_to_public_url(image_path)
    # For testing without a live upload service yet, we can use a known public image URL.
    # This is highly dependent on the image content.
    # Replace this with actual upload logic or a relevant test image URL.
    # For now, we will simulate not being able to get a URL, to avoid breaking flow if not implemented.
    public_image_url = None # Set to a real URL for actual testing
    
    # A more robust approach would be to handle the case where the image cannot be uploaded
    # or made public. For now, we'll log a warning and return no results if no URL.
    if not public_image_url:
        logger.warning(f"VisualProductSearch: Could not get a public URL for image {image_path}. Skipping visual search.")
        # To allow the text search arm to proceed, we don't return an error string here,
        # but an empty list, which is the expected type.
        return []


    try:
        from serpapi import GoogleSearch
        search_params = {
            "engine": "google_lens",
            "url": public_image_url,
            "api_key": serp_api_key,
            "num": 10 # Get up to 10 visual matches
        }
        # Add text query if provided (effectiveness to be tested)
        if query_text:
            search_params["q"] = query_text
            # Consider testing search_type="products" or "visual_matches"
            # search_params["search_type"] = "products" 


        logger.info(f"VisualProductSearch: SerpAPI params: {search_params}")
        search = GoogleSearch(search_params)
        results = search.get_dict()

        if "visual_matches" in results and results["visual_matches"]:
            logger.info(f"VisualProductSearch: Found {len(results['visual_matches'])} visual matches.")
            for i, item in enumerate(results["visual_matches"][:5], 1): # Take top 5
                title = item.get("title", "Unknown Product")
                price_info = item.get("price", {})
                price = price_info.get("value") if isinstance(price_info, dict) else price_info # price can be string or dict
                if price is None : price = "Price not available"

                source = item.get("source", "Unknown Store")
                link = item.get("link", "#")
                
                # Ensure link is a proper URL, fallback to a Google search if not
                if not link or link == "#":
                    link = f"https://www.google.com/search?q={title.replace(' ', '+')}"

                products.append({
                    "title": title,
                    "price": str(price), # Ensure price is string
                    "retailer": source,
                    "url": link,
                    "source": source, # For compatibility
                    "image": item.get("thumbnail", "") # Visual matches often have thumbnails
                })
                logger.info(f"VisualProductSearch: Parsed visual match {i}: {title} - {price} - {source}")
        else:
            logger.warning(f"VisualProductSearch: No 'visual_matches' found in SerpAPI response for {public_image_url}. Keys: {list(results.keys())}")

    except ImportError:
        logger.error("VisualProductSearch: google-search-results package is not installed.")
        # Return empty list, not an error string, to maintain type consistency
        return []
    except Exception as e:
        logger.error(f"VisualProductSearch: Error during visual search: {e}", exc_info=True)
        # Return empty list
        return []
        
    logger.info(f"VisualProductSearch: Returning {len(products)} products.")
    return products

# --- Response Parsing --- #
def parse_agent_response_to_products(agent_response: str) -> list:
    products: list[dict] = []
    if not agent_response or not isinstance(agent_response, str):
        logger.warning("NEW_PARSER: Agent response was empty or not a string.")
        return products

    logger.info(f"NEW_PARSER: Attempting to parse agent response (first 200 chars): {agent_response[:200]}...")
    
    lines = agent_response.split('\n')
    logger.debug(f"NEW_PARSER: Split into {len(lines)} lines.")
    
    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        logger.debug(f"NEW_PARSER: Processing line {line_idx + 1}/{len(lines)}: '{line}'")

        if not line:
            logger.debug(f"NEW_PARSER: Line {line_idx + 1} is empty, skipping.")
            continue
            
        if not (line and len(line) > 0 and line[0].isdigit() and '. ' in line):
            logger.debug(f"NEW_PARSER: Line {line_idx + 1} is not a potential product (no digit/'. '). Skipping: '{line}'")
            continue
            
        logger.info(f"NEW_PARSER: Potential product line identified: '{line}'")
        
        try:
            content_after_number = line.split('. ', 1)[1].strip()
            logger.debug(f"NEW_PARSER: Content after number: '{content_after_number}'")
            
            # Robustly split by " - ", handling potential variations
            # Regex for splitting by " - ", allowing for variations in dashes or spacing if needed in future
            # For now, direct split is fine as per logs
            parts = [p.strip() for p in content_after_number.split(' - ') if p.strip()]
            logger.debug(f"NEW_PARSER: Split into {len(parts)} parts: {parts}")
            
            if len(parts) == 0:
                logger.warning(f"NEW_PARSER: No parts found after splitting content: '{content_after_number}'")
                continue
            
            # Title is always the first part
            title = parts[0]
            price = "Price not available"
            retailer = "Unknown Store"
            url = "#"
            
            # Iterate through remaining parts to find price, store, link
            # This is more flexible than assuming fixed positions
            for part_idx, part_content in enumerate(parts[1:]):
                logger.debug(f"NEW_PARSER: Analyzing part {part_idx + 1} of {len(parts)-1}: '{part_content}'")
                part_lower = part_content.lower()

                if part_lower.startswith("price:"):
                    price = part_content.split(':', 1)[-1].strip()
                    logger.debug(f"NEW_PARSER: Found price (keyword): '{price}'")
                elif part_content.startswith("$") and price == "Price not available": # Avoid overwriting if already found by keyword
                    price = part_content
                    logger.debug(f"NEW_PARSER: Found price (starts with $): '{price}'")
                elif part_lower.startswith("store:"):
                    retailer = part_content.split(':', 1)[-1].strip()
                    logger.debug(f"NEW_PARSER: Found retailer (keyword): '{retailer}'")
                elif part_lower.startswith("link:"):
                    url = part_content.split(':', 1)[-1].strip()
                    logger.debug(f"NEW_PARSER: Found URL (keyword): '{url}'")
                elif retailer == "Unknown Store": # If store not found by keyword, take the first non-price/non-link part as potential store
                    if not (part_content.startswith("$") or part_lower.startswith("http")):
                        # Heuristic: if it contains common store names, more likely a store
                        common_stores = ['amazon', 'walmart', 'target', 'home depot', 'wayfair', 'ikea', 'ashley']
                        if any(store_name in part_lower for store_name in common_stores) or len(parts) <= 3: # If few parts, more likely a store name
                            retailer = part_content
                            logger.debug(f"NEW_PARSER: Found retailer (heuristic): '{retailer}'")
            
            if url == "#" and title and title != "Unknown Product":
                url = f"https://www.google.com/search?tbm=shop&q={title.replace(' ', '+')}"
                logger.debug(f"NEW_PARSER: Generated fallback URL: '{url}'")

            product = {
                "title": title,
                "price": price,
                "retailer": retailer,
                "url": url,
                "source": retailer, # For compatibility
                "image": ""
            }
            products.append(product)
            logger.info(f"NEW_PARSER: Successfully parsed and added product: {product}")
            
        except Exception as e:
            logger.error(f"NEW_PARSER: Error parsing line '{line}': {e}", exc_info=True)
            continue # Crucial to continue to next line
    
    logger.info(f"NEW_PARSER: FINAL RESULT - Total products extracted: {len(products)}")
    return products

# Global variable to store latest tool results
_latest_tool_results = []

def create_function_calling_agent():
    """
    Creates a new function calling agent for product searching.
    This is a more reliable alternative to the REACT agent.
    """
    logger.info("Creating function calling agent...")
    
    # Ensure keys are loaded from config
    openai_key = OPENAI_API_KEY
    serp_key = SERP_API_KEY
    
    if not openai_key:
        logger.error("OpenAI API key not found. Function calling agent creation failed.")
        return None
    
    if not serp_key:
        logger.error("SerpAPI key not found. Function calling agent creation failed.")
        return None
    
    try:
        from langchain.tools import tool
        
        @tool
        def search_products(query: str, max_results: int = 10, enhance_query: bool = True) -> str:
            """Search for furniture and home decor products on Google Shopping.
            
            Args:
                query: Search query for furniture products. Include style, material, color, and type
                max_results: Maximum number of results to return (default: 10)
                enhance_query: Whether to enhance the query with furniture-specific terms (default: True)
            
            Returns:
                JSON string containing product search results
            """
            global _latest_tool_results
            logger.info(f"Executing search_products tool with query: '{query}'")
            
            try:
                # Use cached search
                from utils.cache import cached_product_search
                search_results = cached_product_search(query)
                
                # Parse results
                products = parse_agent_response_to_products(search_results)
                
                # Apply filters
                from utils.filters import filter_and_validate_results
                filtered_products = filter_and_validate_results(products)
                
                # Limit results
                final_products = filtered_products[:max_results]
                
                # Store results globally for access outside the tool
                _latest_tool_results = final_products
                
                logger.info(f"Tool returned {len(final_products)} products")
                
                # Return as JSON string for the agent
                import json
                return json.dumps({
                    "status": "success",
                    "products": final_products,
                    "count": len(final_products)
                })
                
            except Exception as e:
                logger.error(f"Error in search_products tool: {e}")
                _latest_tool_results = []
                return json.dumps({
                    "status": "error",
                    "message": str(e),
                    "products": [],
                    "count": 0
                })
        
        # Create the LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_key
        )

        # Create agent with the tool
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a furniture shopping assistant. Use the search_products tool to find furniture items for users. Always call the tool with the user's query."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the agent
        agent = create_tool_calling_agent(llm, [search_products], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[search_products], verbose=True)
        
        logger.info("✅ Function calling agent created successfully")
        return agent_executor
        
    except Exception as e:
        logger.error(f"❌ Failed to create function calling agent: {e}", exc_info=True)
        return None

def search_with_function_calling_agent(query: str, style_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Search for products using the function calling agent.
    
    Args:
        query: Base search query
        style_info: Optional style information to enhance query
    
    Returns:
        List of product dictionaries
    """
    logger.info(f"Searching with function calling agent: '{query}'")
    
    # Create agent
    agent = create_function_calling_agent()
    if not agent:
        logger.error("Failed to create function calling agent")
        return []
    
    # Enhance query if style info is provided
    if style_info:
        from utils.filters import enhance_furniture_query
        enhanced_query = enhance_furniture_query(query, style_info)
    else:
        enhanced_query = query
    
    try:
        # Store the results globally so our tool can return them
        global _latest_tool_results
        _latest_tool_results = []
        
        # Invoke the agent
        response = agent.invoke({"input": f"Search for furniture products matching: {enhanced_query}"})
        
        # Check if our tool stored results
        if _latest_tool_results:
            logger.info(f"Function calling agent returned {len(_latest_tool_results)} products")
            return _latest_tool_results
        else:
            logger.warning("No tool results found")
            return []
        
    except Exception as e:
        logger.error(f"Error in function calling search: {e}", exc_info=True)
        return []

def search_products_enhanced(query: str, 
                           style_info: Optional[Dict[str, Any]] = None,
                           query_embedding: Optional[np.ndarray] = None,
                           use_function_calling: bool = True,
                           user_budget_hint: Optional[float] = None,
                           search_method: str = 'text_only', # NEW: 'text_only' or 'hybrid'
                           image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Enhanced product search that integrates caching, filtering, and hybrid ranking.
    It can now route between text-only search and a hybrid visual+text search.
    
    Args:
        query: Base search query.
        style_info: Optional style information to enhance query.
        query_embedding: Optional CLIP embedding for visual similarity ranking.
        use_function_calling: Whether to use the function calling agent for text search.
        user_budget_hint: Optional user budget preference for price filtering.
        search_method: The search strategy to use ('text_only' or 'hybrid').
        image_path: The path to the image crop, required for 'hybrid' search.
    
    Returns:
        List of filtered and ranked product dictionaries.
    """
    logger.info(f"Enhanced search started for '{query}' with method: {search_method}")
    
    products = []
    try:
        # Step 1: Choose search strategy based on the method
        if search_method == 'hybrid' and image_path and ENABLE_REVERSE_IMAGE_SEARCH:
            logger.info(f"Using Hybrid Search for image: {image_path}")
            # Use the more powerful reverse image search as the primary source
            from utils.cache import cached_reverse_image_search
            
            enhanced_query = query
            if style_info:
                from utils.filters import enhance_furniture_query
                enhanced_query = enhance_furniture_query(query, style_info)
            
            # This function performs a reverse image search via Google Lens
            # It's cached to avoid re-uploading and re-searching the same image
            search_results_json = cached_reverse_image_search(image_path, enhanced_query)
            
            if search_results_json:
                # The response from reverse image search is already a list of dicts
                products = search_results_json
            else:
                logger.warning("Reverse image search returned no results.")
                products = []

        else:
            if search_method == 'hybrid':
                logger.warning("Hybrid search requested but not possible. Check image_path and .env settings. Falling back to text search.")
            
            # Fallback to text-based search (either agent or older methods)
            if use_function_calling:
                logger.info("Using function calling agent for text search.")
                products = search_with_function_calling_agent(query, style_info)
            else:
                logger.info("Using legacy REACT agent for text search.")
                if style_info:
                    from utils.filters import enhance_furniture_query
                    enhanced_query = enhance_furniture_query(query, style_info)
                else:
                    enhanced_query = query
                
                from utils.cache import cached_product_search
                search_results = cached_product_search(enhanced_query)
                products = parse_agent_response_to_products(search_results)
        
        if not products:
            logger.warning("Initial search phase returned no products.")
            return []
        
        logger.info(f"Found {len(products)} initial products from '{search_method}' search.")
        
        # Step 2: Apply filters
        from utils.filters import filter_and_validate_results
        filtered_products = filter_and_validate_results(products, user_budget_hint)
        
        if not filtered_products:
            logger.warning("No products remained after filtering")
            return []
        
        logger.info(f"After filtering: {len(filtered_products)} products")
        
        # Step 3: Apply hybrid ranking if embedding is available
        if query_embedding is not None:
            from utils.hybrid_ranking import hybrid_rank_results
            ranked_products = hybrid_rank_results(filtered_products, query_embedding)
            logger.info(f"Applied hybrid ranking to {len(ranked_products)} products")
        else:
            ranked_products = filtered_products
            logger.info("No query embedding available, using original ranking")
        
        # Step 4: Add metadata
        for i, product in enumerate(ranked_products):
            product['search_rank'] = i + 1
            product['query'] = query
            if style_info:
                product['style_info'] = style_info
        
        logger.info(f"Enhanced search completed: {len(ranked_products)} final products")
        return ranked_products
        
    except Exception as e:
        logger.error(f"Error in enhanced product search: {e}", exc_info=True)
        return []

def search_products_with_visual_similarity(image_path: str,
                                         caption_data: Dict[str, Any],
                                         query_embedding: Optional[np.ndarray] = None,
                                         user_budget_hint: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Search for products using visual similarity and caption data.
    
    Args:
        image_path: Path to the furniture crop image
        caption_data: Caption data with style information
        query_embedding: CLIP embedding of the query image
        user_budget_hint: Optional user budget preference
    
    Returns:
        List of filtered and ranked product dictionaries
    """
    logger.info(f"Visual similarity search for: {image_path}")
    
    # Extract base query from caption
    base_query = caption_data.get('caption', 'furniture')
    
    # Extract style information
    style_info = {
        'style': caption_data.get('style'),
        'material': caption_data.get('material'),
        'colour': caption_data.get('colour'),
        'era': caption_data.get('era')
    }
    
    # Remove None values
    style_info = {k: v for k, v in style_info.items() if v}
    
    # Use enhanced search
    products = search_products_enhanced(
        query=base_query,
        style_info=style_info,
        query_embedding=query_embedding,
        use_function_calling=True,
        user_budget_hint=user_budget_hint
    )
    
    # Add image metadata
    for product in products:
        product['source_image'] = image_path
        product['caption_data'] = caption_data
    
    return products

if __name__ == '__main__':
    print("--- Product Matcher Agent Test ---")
    if not OPENAI_API_KEY or not SERP_API_KEY:
        print("❌ OPENAI_API_KEY and/or SERP_API_KEY are not set in .env file. Aborting test.")
        print("Please create a .env file in the interior_designer/ directory with your keys:")
        print("OPENAI_API_KEY='sk-...'\nSERP_API_KEY='your_serp_api_key...'")
    else:
        print("API keys appear to be loaded. Creating agent...")
        product_agent = create_new_product_search_agent()
        if product_agent:
            print("✅ Agent created. Testing with a sample query...")
            test_query = "Find a modern minimalist armchair in grey"
            try:
                response = product_agent.invoke({"input": test_query})
                print(f"\nRaw Agent Response for query '{test_query}':\n{response}")
                
                if isinstance(response, dict) and 'output' in response:
                    parsed = parse_agent_response_to_products(response['output'])
                    print(f"\nParsed Products ({len(parsed)}):")
                    for p in parsed:
                        print(f"  - {p['title']} | {p['price']} | {p['retailer']} | {p['url']}")
                else:
                    print("\nAgent response was not in the expected dictionary format.")

            except Exception as e:
                print(f"❌ Error during agent test query: {e}")
        else:
            print("❌ Agent creation failed. Check logs for details.")
    print("----------------------------------") 