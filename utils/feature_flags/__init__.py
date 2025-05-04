"""
Feature flag system to control optional functionality.
"""
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

def is_enabled(flag_name, default=False):
    """
    Check if a feature flag is enabled via environment variables.
    
    Args:
        flag_name: Name of the feature flag to check
        default: Default value if flag is not set in environment
        
    Returns:
        bool: True if feature is enabled, False otherwise
    """
    # Get the value from environment, defaulting to "off"
    env_value = os.getenv(flag_name, "off").lower()
    
    # Check if it's a "truthy" value
    return env_value in ("on", "true", "1", "yes")
    
# Define flags used in the application
# These represent the flags supported by the application
FLAGS = {
    # Dual model inference flags
    "FINETUNE_MODEL": False,  # Enable fine-tuned model
    
    # Search and ranking flags
    "VECTOR_SEARCH": False,   # Enable vector-based search
    "LLM_RERANK": False,      # Use LLM to rerank results
    
    # Link verification flags
    "HEADLESS_VERIFY": False, # Use headless browser verification
    
    # Infrastructure flags
    "EXT_CACHE": False,       # Enable external caching (Redis/S3)
    "METRICS": False,         # Enable metrics collection
    
    # UI flags
    "PROGRESSIVE_CARDS": True, # Show progressive loading of cards (default on)
    
    # Advanced features
    "AR_SDK": False,          # AR mode for snap-to-shop
    "DESIGN_SUGGESTIONS": False, # Show design suggestions
    "AFFILIATE_TAGS": False,  # Add affiliate tags to links
    
    # New product search features
    "SERPAPI_REVIMG": True,   # Use SerpAPI for true reverse image search
    "GPT4O_CAPTIONS": True,   # Use GPT-4o for rich captions
    "COLOR_EXTRACTOR": True,  # Extract colors from images using k-means
    "MATERIAL_DETECTOR": True, # Detect materials in images using GPT-4o
    "SSIM_PRODUCT_VALIDATION": True, # Use SSIM to validate product matches
    "NEW_PRODUCT_RANKER": True, # Use improved ranking algorithm
    "PARALLEL_CAPTION_SEARCH": True, # Search with multiple captions in parallel
} 