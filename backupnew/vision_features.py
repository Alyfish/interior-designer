"""interior_designer.vision_features

Centralised helpers for visual feature extraction (CLIP) and captioning (BLIP).
These utilities are imported by Streamlit and product-matching code to enable
visual similarity search and richer text queries.

The module is designed to:
• lazily download / load models on first use only;
• cache heavy objects with functools.lru_cache so multiple calls are cheap;
• fall back gracefully if the required packages (open_clip_torch, salesforce-lavis)
  are missing – returning stub values without crashing the app.
"""
from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache
from typing import Tuple
import os

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover – torch is in venv but guard anyway
    torch = None  # type: ignore

# Import transformers at the top level, as it will be the primary BLIP provider
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None

import json # Added for JSON parsing
try:
    from PIL import Image # Moved PIL import here for wider use
except ImportError: # pragma: no cover
    Image = None # type: ignore

# For OpenAI Polishing (optional - prepare import)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# For GPT-4 Vision (much better than BLIP)
try:
    from openai import OpenAI
    GPT4V_AVAILABLE = True
except ImportError:
    OpenAI = None
    GPT4V_AVAILABLE = False

# -------------------------------------------------------------
# Optional heavy deps. We import inside functions so the app can
# still start if they are absent. Users will simply get a warning
# and visual features will be disabled.
# -------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------- Model loading helpers ----------

@lru_cache(maxsize=1)
def _load_clip() -> Tuple["torch.nn.Module", "callable"]:  # type: ignore[name-defined]
    """Lazy-load Open-CLIP (ViT-B/32 by default).

    Returns
    -------
    model : torch.nn.Module
    preprocess : Callable[[PIL.Image.Image], torch.Tensor]
    """
    if torch is None:
        raise RuntimeError("PyTorch is not available. CLIP features disabled.")

    try:
        import open_clip
    except Exception as e:
        raise RuntimeError("open_clip_torch not installed – run `pip install open_clip_torch`." ) from e

    # Use CPU or GPU automatically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    logger.info("✅ Open-CLIP model loaded (%s)", device)
    return model, preprocess


@lru_cache(maxsize=1)
def _load_blip_transformer() -> Tuple["BlipForConditionalGeneration", "BlipProcessor"]:
    """Lazy-load BLIP model via Hugging Face Transformers."""
    if torch is None or BlipProcessor is None or BlipForConditionalGeneration is None:
        raise RuntimeError("PyTorch or Transformers BLIP components not available. BLIP captions disabled.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Using a more general base model, specific prompting will guide it.
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        model.eval()
        logger.info("✅ Transformers BLIP model loaded (Salesforce/blip-image-captioning-base, %s)", device)
        return model, processor
    except Exception as e:
        raise RuntimeError("Failed to load Transformers BLIP model.") from e


# ---------- Public API ----------

def extract_clip_embedding(image_path: str | Path) -> np.ndarray:
    """Return a normalised CLIP embedding for the image (dtype float32)."""
    if torch is None:
        logger.warning("PyTorch not found – returning zero embedding.")
        return np.zeros(512, dtype="float32")

    model, preprocess = _load_clip()
    from PIL import Image  # local import to avoid headless issues if PIL missing

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # type: ignore[arg-type]
    device = next(model.parameters()).device
    with torch.no_grad():
        embedding = model.encode_image(tensor.to(device))  # type: ignore[attr-defined]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().float().numpy()


def generate_blip_caption(image_path: str | Path) -> str:
    """Generate a descriptive caption for the image using Transformers BLIP.
       This function is now a fallback or can be used for generic captioning.
       For designer-style JSON captions, use get_caption_json directly.
    """
    try:
        # For a simple, direct caption, we don't use the DESIGNER_PROMPT here.
        # This maintains its old behavior if called directly.
        # We can use a generic prompt or let BLIP do its default.
        if Image is None: raise ImportError("PIL not found")
        pil_img = Image.open(image_path).convert("RGB")
        # Using a simpler prompt for this legacy function for now
        generic_prompt = "a photo of" 
        caption = _caption_image_raw(pil_img, prompt=generic_prompt, max_tokens=50)
        
        # Basic cleaning
        if caption.lower().startswith(generic_prompt):
            caption = caption[len(generic_prompt):].strip()

        logger.info(f"Generated generic BLIP caption for {image_path}: {caption}")
        return caption

    except Exception as e: # Broad exception to catch PIL errors, model errors etc.
        logger.warning("Generic BLIP caption (generate_blip_caption) unavailable: %s", e)
        return ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (returns 0.0 on error)."""
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape or a.ndim != 1:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom) 

# --- Constants ---
# Simplified prompts that work better with BLIP
SIMPLE_DESIGNER_PROMPT = "A detailed description of this furniture piece, including its style, material, and color:"
SIMPLE_ROOM_PROMPT = "The interior design style of this room:"

# --- GPT-4 Vision prompts (much more effective than BLIP)
GPT4V_DESIGNER_PROMPT = """
You are an expert interior designer. Analyze the furniture piece in the image.
Respond ONLY with a valid JSON object. Do not include any other text, greetings, or explanations.
The JSON object must contain these exact keys: "style", "material", "colour", "era", and "caption".
The "caption" value must be a detailed, single-sentence description of the furniture piece, perfect for use as a search query.
Example response format:
{
  "style": "mid-century modern",
  "material": "walnut wood and fabric",
  "colour": "dark brown and beige",
  "era": "1960s",
  "caption": "A mid-century modern accent chair with a dark brown walnut frame and beige fabric upholstery, featuring clean lines and tapered legs."
}
"""

# GPT4V_ROOM_PROMPT = "Identify the interior design style of this room in 5-10 words (e.g., 'modern minimalist', 'traditional cozy', 'industrial loft'):" # No longer used

# --- Helper Functions ---
def _extract_design_info_from_caption(caption: str) -> dict:
    """
    Extract design information from a natural language caption.
    Uses simple keyword matching to identify style, material, color, era.
    """
    caption_lower = caption.lower()
    
    # Common style keywords
    style_keywords = {
        'modern': 'modern', 'contemporary': 'contemporary', 'traditional': 'traditional',
        'rustic': 'rustic', 'industrial': 'industrial', 'scandinavian': 'scandinavian',
        'mid-century': 'mid-century', 'vintage': 'vintage', 'classic': 'classic',
        'minimalist': 'minimalist', 'bohemian': 'bohemian', 'farmhouse': 'farmhouse'
    }
    
    # Common material keywords
    material_keywords = {
        'wood': 'wood', 'wooden': 'wood', 'oak': 'oak', 'pine': 'pine', 'mahogany': 'mahogany',
        'metal': 'metal', 'steel': 'steel', 'iron': 'iron', 'aluminum': 'aluminum',
        'fabric': 'fabric', 'leather': 'leather', 'plastic': 'plastic', 'glass': 'glass',
        'velvet': 'velvet', 'cotton': 'cotton', 'linen': 'linen'
    }
    
    # Common color keywords
    color_keywords = {
        'black': 'black', 'white': 'white', 'brown': 'brown', 'gray': 'gray', 'grey': 'gray',
        'blue': 'blue', 'red': 'red', 'green': 'green', 'yellow': 'yellow', 'orange': 'orange',
        'purple': 'purple', 'pink': 'pink', 'beige': 'beige', 'cream': 'cream', 'tan': 'tan'
    }
    
    # Era keywords
    era_keywords = {
        '1950s': '1950s', '1960s': '1960s', '1970s': '1970s', '1980s': '1980s', '1990s': '1990s',
        'antique': 'antique', 'vintage': 'vintage', 'retro': 'retro', 'contemporary': 'contemporary'
    }
    
    # Extract information
    detected_style = next((style for keyword, style in style_keywords.items() if keyword in caption_lower), 'unknown')
    detected_material = next((material for keyword, material in material_keywords.items() if keyword in caption_lower), 'unknown')
    detected_color = next((color for keyword, color in color_keywords.items() if keyword in caption_lower), 'unknown')
    detected_era = next((era for keyword, era in era_keywords.items() if keyword in caption_lower), 'unknown')
    
    return {
        'style': detected_style,
        'material': detected_material,
        'colour': detected_color,
        'era': detected_era,
        'caption': caption
    }

# --- Helper for raw captioning ---
def _caption_image_raw(img_pil: Image.Image, prompt: str, max_tokens: int = 60) -> str:
    """
    Helper function to get raw caption from BLIP model.
    Takes a PIL image.
    """
    if Image is None:
        logger.warning("PIL (Pillow) not found - cannot process image for captioning.")
        return ""
    try:
        model, processor = _load_blip_transformer()
    except RuntimeError as e:
        logger.warning("BLIP caption (Transformers) unavailable: %s", e)
        return ""

    device = next(model.parameters()).device
    inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens, num_beams=5, early_stopping=True)
    
    raw_caption = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    logger.warning(f"BLIP raw caption: {raw_caption}")
    
    return raw_caption

# --- New Captioning Functions ---

def get_caption_json(image_input: str | Path | np.ndarray, designer: bool = True) -> dict:
    """
    Generate structured caption data from an image.
    
    Now supports GPT-4 Vision for much better accuracy when OpenAI API key is available.
    Falls back to BLIP if GPT-4V is not available.
    """
    # Check if we have OpenAI API key and GPT-4V is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if GPT4V_AVAILABLE and openai_api_key:
        logger.info("Using GPT-4 Vision for caption generation (much more accurate than BLIP)")
        return get_caption_json_gpt4v(image_input, designer, openai_api_key)
    else:
        logger.info("Using BLIP for caption generation (consider adding OPENAI_API_KEY for better accuracy)")
        
        # Original BLIP-based logic
        try:
            if isinstance(image_input, (str, Path)):
                if not os.path.exists(image_input):
                    logger.warning(f"Image file not found: {image_input}")
                    return {"style": "unknown", "material": "unknown", "colour": "unknown", "era": "unknown", "caption": "Image file not found"}
                
                # Load the image using PIL for BLIP
                img_pil = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                # Convert OpenCV image (BGR) to PIL (RGB)
                if cv2 is not None:
                    rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback: assume BGR and convert manually
                    rgb_array = image_input[:, :, ::-1] if len(image_input.shape) == 3 else image_input
                
                img_pil = Image.fromarray(rgb_array.astype(np.uint8))
            else:
                logger.warning(f"Unsupported image input type: {type(image_input)}")
                return {"style": "unknown", "material": "unknown", "colour": "unknown", "era": "unknown", "caption": "Unsupported image format"}

            # Choose appropriate prompt
            prompt = SIMPLE_DESIGNER_PROMPT if designer else SIMPLE_ROOM_PROMPT
            
            # Generate raw caption using BLIP
            raw_caption = _caption_image_raw(img_pil, prompt, max_tokens=60)
            
            # Extract structured information from the caption
            result = _extract_design_info_from_caption(raw_caption)
            
            return result

        except Exception as e:
            logger.error(f"Error in caption generation: {e}")
            return {"style": "unknown", "material": "unknown", "colour": "unknown", "era": "unknown", "caption": "Error generating caption"}


def merge_captions(crop_json: dict, room_style_caption: str) -> dict:
    """
    DEPRECATED: Merges the crop's JSON data with the room style description.
    This is no longer used as we are focusing on per-object characteristics.
    """
    logger.info("merge_captions is deprecated and should no longer be called.")
    return crop_json # Return the original JSON without modification


# --- Optional OpenAI Polishing ---
def polish_caption_with_openai(caption: str, openai_api_key: str) -> str:
    """
    Uses OpenAI to paraphrase and polish a caption. (Untested, conceptual)
    """
    if OpenAI is None:
        logger.warning("OpenAI SDK not installed. Cannot polish caption.")
        return caption
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not available. Cannot polish caption.")
        return caption

    try:
        client = OpenAI(api_key=openai_api_key)
        polish_prompt = (
            "Rewrite this designer object description in one concise sentence, keeping all factual details regarding style, material, color, and era: "
            f"\\n\\n'{caption}'"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or a newer/cheaper model if suitable
            messages=[{"role": "user", "content": polish_prompt}],
            temperature=0.5,
            max_tokens=100 
        )
        polished_caption = response.choices[0].message.content.strip()
        logger.info(f"Polished caption: {polished_caption}")
        return polished_caption
    except Exception as e:
        logger.error(f"Error polishing caption with OpenAI: {e}")
        return caption # Return original caption on error 

# --- GPT-4 Vision Functions ---
def get_caption_json_gpt4v(image_input: str | Path | np.ndarray, designer: bool = True, openai_api_key: str = None) -> dict:
    """
    Use GPT-4 Vision for much more accurate furniture description.
    This is significantly better than BLIP for product matching.
    """
    if not GPT4V_AVAILABLE or not openai_api_key:
        logger.warning("GPT-4 Vision not available, falling back to BLIP")
        return get_caption_json(image_input, designer)
    
    try:
        # Convert image to base64 for GPT-4V
        import base64
        from io import BytesIO
        
        if isinstance(image_input, (str, Path)):
            if not os.path.exists(image_input):
                logger.warning(f"Image file not found: {image_input}")
                return {"style": "unknown", "material": "unknown", "colour": "unknown", "era": "unknown", "caption": "Image file not found"}
            
            with open(image_input, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to base64
            if cv2 is not None:
                # Convert BGR to RGB
                rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                rgb_array = image_input[:, :, ::-1] if len(image_input.shape) == 3 else image_input
            
            pil_image = Image.fromarray(rgb_array.astype(np.uint8))
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            logger.warning(f"Unsupported image input type: {type(image_input)}")
            return {"style": "unknown", "material": "unknown", "colour": "unknown", "era": "unknown", "caption": "Unsupported image format"}

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Choose prompt based on designer flag. Since we deprecated room style, we always use the designer prompt.
        prompt = GPT4V_DESIGNER_PROMPT
        
        # Make GPT-4V request
        response = client.chat.completions.create(
            model="gpt-4o",  # Latest vision model
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        gpt4v_response = response.choices[0].message.content
        logger.info(f"GPT-4V response: {gpt4v_response}")
        
        # We now assume we always want the detailed JSON for an object.
        # The room-style logic path is no longer used from the streamlit app.
        try:
            # Extract JSON from response
            import json
            # Clean the response to better isolate the JSON object
            json_str = gpt4v_response[gpt4v_response.find('{'):gpt4v_response.rfind('}')+1]
            result = json.loads(json_str)
            
            # Ensure all required keys exist, providing sensible defaults if not.
            required_keys = ['style', 'material', 'colour', 'era', 'caption']
            for key in required_keys:
                if key not in result or not result[key]:
                    result[key] = 'unknown'
            
            logger.info(f"GPT-4V parsed JSON: {result}")
            return result

        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse GPT-4V JSON. Using the raw response as the caption.")
            # If JSON parsing fails, use the entire response as the caption. This is a robust fallback.
            return {
                "style": "unknown",
                "material": "unknown",
                "colour": "unknown",
                "era": "unknown",
                "caption": gpt4v_response.strip()
            }
            
    except Exception as e:
        logger.error(f"Error with GPT-4 Vision: {e}", exc_info=True)
        # Fallback to a simple error dictionary to prevent recursion
        return {"style": "error", "material": "error", "colour": "error", "era": "error", "caption": "Error during GPT-4 Vision processing."} 