import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import logging

# Initialize logging for this module
logger = logging.getLogger(__name__)
# Basic config if no handlers are present (e.g. when run directly)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Determine the project root directory dynamically
# This assumes config.py is in interior_designer/
PROJECT_ROOT = Path(__file__).resolve().parent
# Primary env paths to check (trying both with and without dot)
dotenv_path_primary = PROJECT_ROOT / ".env"
env_path_without_dot = PROJECT_ROOT / "env"

# Attempt to load environment variables from .env file
logger.info(f"Attempting to load .env file from primary path: {dotenv_path_primary}")
loaded_primary = load_dotenv(dotenv_path=dotenv_path_primary, override=True, encoding='utf-8')

# If .env not found, try env (without dot)
if not loaded_primary and env_path_without_dot.exists():
    logger.info(f"Trying env file without dot: {env_path_without_dot}")
    loaded_primary = load_dotenv(dotenv_path=env_path_without_dot, override=True, encoding='utf-8')

if loaded_primary:
    # Determine which file was actually loaded
    if dotenv_path_primary.exists():
        logger.info(f"Successfully loaded .env file from: {dotenv_path_primary}")
    else:
        logger.info(f"Successfully loaded env file from: {env_path_without_dot}")
else:
    logger.info(f".env file not found or failed to load from {dotenv_path_primary}. Attempting to find .env in parent directories...")
    # find_dotenv searches from CWD upwards.
    # If running scripts from project root, CWD might be project root.
    # If running from interior_designer, CWD is interior_designer, so it will search interior_designer then parent.
    dotenv_path_found = find_dotenv(filename='.env', usecwd=True, raise_error_if_not_found=False)
    if dotenv_path_found and os.path.exists(dotenv_path_found):
        logger.info(f"Found .env file at: {dotenv_path_found}. Attempting to load.")
        if load_dotenv(dotenv_path=dotenv_path_found, override=True, encoding='utf-8'):
            logger.info(f"Successfully loaded .env file from: {dotenv_path_found}")
        else:
            logger.warning(f"Failed to load .env file found at: {dotenv_path_found}")
    elif not os.environ.get("IS_TEST_ENVIRONMENT"): # Avoid warning in test environments
        logger.warning(f".env file not found using primary path or by searching parent directories. Relying on system environment variables.")

# Essential API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# API Keys
SERP_API_KEY = os.getenv("SERP_API_KEY", "")  # For both Google Shopping text search and Google Lens visual search via SerpAPI
# For backward compatibility, alias SEARCH_API_KEY to the same value
SEARCH_API_KEY = SERP_API_KEY

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "")

# Replicate API key for Mask2Former
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

# Feature flags
ENABLE_REVERSE_IMAGE_SEARCH = os.getenv("REVERSE_IMAGE_SEARCH", "off").lower() in ["on", "true", "1", "yes"]

# Smart Search Feature Flags (ADDITIVE - does not affect existing functionality)
USE_SMART_PRODUCT_SEARCH = os.getenv("USE_SMART_PRODUCT_SEARCH", "false").lower() in ["true", "1", "yes", "on"]
SMART_SEARCH_TIMEOUT = int(os.getenv("SMART_SEARCH_TIMEOUT", "30"))
SMART_SEARCH_MAX_RESULTS = int(os.getenv("SMART_SEARCH_MAX_RESULTS", "10"))
SMART_SEARCH_PRICE_ANALYSIS = os.getenv("SMART_SEARCH_PRICE_ANALYSIS", "true").lower() in ["true", "1", "yes", "on"]
SMART_SEARCH_STYLE_MATCHING = os.getenv("SMART_SEARCH_STYLE_MATCHING", "true").lower() in ["true", "1", "yes", "on"]
SMART_SEARCH_DEBUG = os.getenv("SMART_SEARCH_DEBUG", "false").lower() in ["true", "1", "yes", "on"]

# You can add other keys here as needed, e.g.:
# AMAZON_RAPID_API_KEY = os.getenv("AMAZON_RAPID_KEY")
# BING_API_KEY = os.getenv("BING_KEY")

def check_api_keys():
    """Check if essential API keys are loaded and provide helpful warnings."""
    missing_keys = []
    recommended_keys_for_visual_search = []
    
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY (required for OpenAI GPT models)")
    
    if not SERP_API_KEY:
        missing_keys.append("SERP_API_KEY (for both Google Shopping text search and Google Lens visual search)")
    
    enable_reverse_search_flag = os.getenv("REVERSE_IMAGE_SEARCH", "off").lower() in ["on", "true", "1", "yes"]

    if enable_reverse_search_flag:
        if not IMGBB_API_KEY:
            missing_keys.append("IMGBB_API_KEY (for image uploads for reverse image search)")
    else:
        if not IMGBB_API_KEY:
            recommended_keys_for_visual_search.append("IMGBB_API_KEY (for image uploads for reverse image search)")

    if missing_keys:
        logger.warning(f"Missing required API keys: {', '.join(missing_keys)}")
        logger.warning("Please add them to your .env file (e.g., in the project root or interior_designer/ directory):")
        for key in missing_keys:
            key_name = key.split(" (")[0] # Get the base key name for the .env example
            logger.warning(f"  {key_name}=your_key_here")
        logger.warning("Note: Some features may not work without these keys.")
    
    if recommended_keys_for_visual_search:
        logger.info(f"ℹ️ Info: For full reverse image search functionality, consider setting these optional API keys in your .env file:")
        for key_desc in recommended_keys_for_visual_search:
            key_name = key_desc.split(" (")[0]
            logger.info(f"  {key_name}=your_key_here")
        if "IMGBB_API_KEY" in [k.split(" (")[0] for k in recommended_keys_for_visual_search]:
             logger.info("  Get a free ImgBB key at https://api.imgbb.com/")
        if "SEARCH_API_KEY" in [k.split(" (")[0] for k in recommended_keys_for_visual_search]:
            logger.info("  Get a SearchAPI.io key at https://www.searchapi.io/ (ensure it's stored as SEARCH_API_KEY in .env)")


    # Return true if all *strictly required* keys are present.
    # If reverse search is off, IMGBB_API_KEY is not strictly required.
    all_strictly_required_present = True
    if not OPENAI_API_KEY or not SERP_API_KEY:
        all_strictly_required_present = False
    if enable_reverse_search_flag and not IMGBB_API_KEY:
        all_strictly_required_present = False
        
    return all_strictly_required_present

if __name__ == "__main__":
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger.info("--- Configuration Check (config.py) ---")
    logger.info(f"Project Root (derived in config.py): {PROJECT_ROOT}")
    logger.info(f"Primary .env path checked: {dotenv_path_primary}")
    
    # Re-check loading status for output
    env_vars_loaded = bool(SERP_API_KEY or IMGBB_API_KEY or SEARCH_API_KEY) # Simple check if any key is loaded
    if env_vars_loaded:
        logger.info(".env variables seem to be loaded (at least one key is present).")
    else:
        logger.warning(".env variables do not seem to be loaded (no relevant API keys found in environment).")

    check_api_keys() # This will now use logger
    logger.info(f"OpenAI API Key Loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
    logger.info(f"SerpAPI Key (for text + visual search) Loaded: {'Yes' if SERP_API_KEY else 'No'}")
    logger.info(f"ImgBB Key Loaded: {'Yes' if IMGBB_API_KEY else 'No'}")
    reverse_search_flag = os.getenv("REVERSE_IMAGE_SEARCH", "off").lower() in ["on", "true", "1", "yes"]
    logger.info(f"REVERSE_IMAGE_SEARCH flag: {reverse_search_flag}")
    logger.info("-------------------------------------") 