#!/usr/bin/env python3
"""
🏡 AI Interior Designer with Furniture Object Detection and Product Matching

This app enables users to:
1. Upload interior images  
2. Auto-detect furniture using YOLOv8-Seg
3. Extract visual features using CLIP and BLIP
4. Search for matching products using LangChain agents and SerpAPI

Architecture:
- YOLOv8-Seg for object detection and segmentation
- CLIP for visual feature embeddings  
- BLIP for image captioning
- GPT-4 Vision for detailed descriptions
- LangChain agents with SerpAPI for product searches

Main components:
- new_object_detector.py: YOLOv8 detection logic
- vision_features.py: CLIP embeddings and BLIP captions  
- new_product_matcher.py: LangChain product search agents
"""

# Suppress warnings before imports
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*bottleneck.*")
warnings.filterwarnings("ignore", message=".*RuntimeError.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --- Utility Function (import-safe) -----------------------------------------
# Placed before heavy Streamlit imports so other modules/tests can import this
# file just to access `get_config_status` without triggering Streamlit logic.

def get_config_status(
    serp_api_key: str,
    imgbb_api_key: str,
    reverse_image_search_enabled: bool,
):
    """Return a dictionary describing UI configuration readiness.

    Keys returned:
        search_mode: str  # 'hybrid_available', 'hybrid_degraded_no_upload',
                          # 'hybrid_degraded_no_visual_api', 'text_only_feature_off',
                          # 'text_only_serp_missing'
        serp_api_ready: bool
        visual_search_api_ready: bool
        image_upload_ready: bool
        reverse_image_search_enabled: bool
    """
    serp_ready = bool(serp_api_key)
    visual_ready = bool(serp_api_key)  # Using same key for visual search now
    upload_ready = bool(imgbb_api_key)

    if not serp_ready:
        mode = "text_only_serp_missing"
    elif not reverse_image_search_enabled:
        mode = "text_only_feature_off"
    else:
        # Feature flag ON, text key present
        if visual_ready and upload_ready:
            mode = "hybrid_available"
        elif not visual_ready:
            mode = "hybrid_degraded_no_visual_api"
        else:  # upload missing
            mode = "hybrid_degraded_no_upload"

    return {
        "search_mode": mode,
        "serp_api_ready": serp_ready,
        "visual_search_api_ready": visual_ready,
        "image_upload_ready": upload_ready,
        "reverse_image_search_enabled": reverse_image_search_enabled,
    }

# ---------------------------------------------------------------------------

# Fix PyTorch/Streamlit compatibility issues
import os
os.environ['TORCH_MULTIPROCESSING_SHARING_STRATEGY'] = 'file_system'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Additional torch settings to prevent event loop issues
try:
    import torch
    if hasattr(torch, 'set_sharing_strategy'):
        torch.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
except ImportError:
    pass

# Standard library imports
import sys
import logging
import tempfile
import json
import time
import base64
import io
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional

# Third-party imports
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Local imports - Updated for new file structure with Mask2Former
from vision_features import generate_blip_caption, extract_clip_embedding, get_caption_json, get_caption_json_gpt4v, merge_captions, polish_caption_with_openai
from object_detector import ObjectDetector
from product_matcher import (
    create_new_product_search_agent, 
    parse_agent_response_to_products, 
    search_products_hybrid, 
    search_products_enhanced,
    search_products_with_visual_similarity,
    ENABLE_REVERSE_IMAGE_SEARCH
)
# This import is no longer needed
# from replicate_detector import ReplicateMask2Former
from config import *
from config import REPLICATE_API_TOKEN

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Run API key check from config when Streamlit app starts
check_api_keys()

# Environment variables are now available from the imports above
# OPENAI_API_KEY and SERP_API_KEY are imported from config
SERP_KEY = SERP_API_KEY  # Rename for consistency with the rest of the code

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Interior Designer",
    page_icon="🏠",
    layout="wide"
)

# --- Custom CSS (Matching Original Style) ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
    }
    .detection-stats {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background-color: #ffffff;
    }
    .product-card img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .product-card h3 {
        font-size: 14px;
        margin: 8px 0 4px 0;
        color: #333;
    }
    .product-card p {
        font-size: 12px;
        margin: 2px 0;
        color: #666;
    }
    .product-card a {
        display: inline-block;
        background-color: #ff4b4b;
        color: white;
        padding: 8px;
        border-radius: 4px;
        text-decoration: none;
        margin-top: auto; /* Pushes button to bottom */
    }
    .object-item {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .object-item:hover {
        border-color: #ff4b4b;
        background-color: #fff5f5;
    }
    .object-item.selected {
        border-color: #ff4b4b;
        background-color: #ffe5e5;
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .canvas-container {
        position: relative;
        margin: 0 auto;
        border: 1px solid #eee;
        overflow: hidden;
    }
    .canvas-layer {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        cursor: pointer;
    }
    .selected-item {
        background-color: #e8f5e8;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .product-price {
        font-size: 1.2em;
        font-weight: bold;
        color: #e74c3c;
    }
    .product-title {
        font-size: 1.1em;
        margin-bottom: 10px;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "image_processed": False,
        "original_image_cv": None,
        "detected_objects": [],
        "segmented_image_path": None,
        "selected_object_indices": [], # Store indices of st.session_state.detected_objects
        "product_search_results": {},
        "ai_product_agent": None,
        "temp_upload_path": None,
        "selected_objects": [], # For compatibility with original app
        'processed': False,
        'image': None,
        'objects': [],
        'temp_img_path': None,
        'object_detector': None,
        'product_agent': None,
        'detection_results': None,
        'interactive_html': None,
        'search_method': "Hybrid Search (Text + Visual)"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Helper Functions ---
@st.cache_resource
def get_object_detector():
    """Initializes and returns the main object detector."""
    return ObjectDetector()

# This function is no longer needed
# @st.cache_resource
# def get_replicate_mask2former_detector():
#     """Initializes and returns the Replicate Mask2Former detector."""
#     return ReplicateMask2Former()

@st.cache_resource
def get_product_agent():
    """Initializes and returns the LangChain product search agent."""
    try:
        agent = create_new_product_search_agent()
        logger.info("✅ Product search agent initialized successfully")
        return agent
    except Exception as e:
        st.error(f"Failed to initialize Product Agent: {e}")
        logger.error(f"❌ Failed to initialize product agent: {e}")
        return None

def create_interactive_html(image, objects, products_data=None):
    """Create sophisticated interactive HTML component for object selection."""
    
    # Log image dimensions being used for the component
    img_height, img_width, _ = image.shape
    logger.info(f"Creating interactive HTML for image with dimensions: {img_width}x{img_height}")

    # Convert image to base64 for embedding
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Calculate display dimensions
    img_height, img_width = image.shape[:2]
    max_width = 1200
    max_height = 800
    scale = min(max_width / img_width, max_height / img_height, 1)
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)
    
    # Color mapping for different object types
    object_colors = {
        'wall': '#8B4513',      # Brown
        'floor': '#DEB887',     # BurlyWood
        'chair': '#DAA520',     # GoldenRod
        'couch': '#BC8F8F',     # RosyBrown
        'sofa': '#D2691E',      # Chocolate
        'bed': '#FA8072',       # Salmon
        'dining table': '#CD853F', # Peru
        'tv': '#1E90FF',        # DodgerBlue
        'potted plant': '#228B22', # ForestGreen
        'custom': '#FF1493',    # DeepPink
    }
    
    # Scale contours for display and prepare objects
    js_objects = []
    for i, obj in enumerate(objects):
        # Scale contours
        scaled_contours = []
        if 'contours' in obj:
            for contour in obj['contours']:
                scaled_contour = [[int(point[0] * scale), int(point[1] * scale)] for point in contour]
                scaled_contours.append(scaled_contour)
        
        # Scale bbox if present
        scaled_bbox = None
        if 'bbox' in obj:
            bbox = obj['bbox']
            scaled_bbox = [
                int(bbox[0] * scale),
                int(bbox[1] * scale),
                int(bbox[2] * scale),
                int(bbox[3] * scale)
            ]
        
        obj_data = {
            'id': obj.get('id', i),
            'class': obj.get('class', 'unknown'),
            'confidence': obj.get('confidence', 0.0),
            'contours': scaled_contours,
            'bbox': scaled_bbox,
            'area': obj.get('area', 0),
            'source': obj.get('source', 'detection'),
        }
        js_objects.append(obj_data)

    # Prepare products data for JavaScript
    js_products_data = {}
    if products_data:
        for obj_id, products in products_data.items():
            js_products_data[str(obj_id)] = products

    # Create the HTML with modal carousel
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                margin: 0;
                padding: 10px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #f8f9fa;
                overflow-x: hidden;
            }}
            .container {{
                max-width: 100%;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .controls {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: center;
                border-bottom: 3px solid #e1e5e9;
            }}
            .btn {{
                background: rgba(255,255,255,0.2);
                color: white;
                border: 2px solid rgba(255,255,255,0.3);
                padding: 8px 16px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 500;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 5px;
            }}
            .btn:hover, .btn.active {{
                background: rgba(255,255,255,0.9);
                color: #333;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .checkbox-container {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-left: auto;
                color: white;
                font-size: 14px;
            }}
            .checkbox-container input[type="checkbox"] {{
                transform: scale(1.2);
            }}
            .canvas-container {{
                position: relative;
                text-align: center;
                background: #fafafa;
                padding: 20px;
            }}
            canvas {{
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                cursor: crosshair;
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .mode-indicator {{
                position: absolute;
                top: 30px;
                left: 30px;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
                z-index: 20;
            }}
            .tooltip {{
                position: absolute;
                background: rgba(0,0,0,0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                pointer-events: none;
                z-index: 30;
                white-space: nowrap;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .selection-summary {{
                position: fixed;
                bottom: 20px;
                left: 20px;
                background: rgba(255,255,255,0.95);
                border: 3px solid #2196F3;
                border-radius: 12px;
                padding: 15px;
                font-size: 13px;
                max-height: 250px;
                max-width: 350px;
                overflow-y: auto;
                z-index: 25;
                box-shadow: 0 8px 25px rgba(0,0,0,0.2);
                font-family: 'Courier New', monospace;
                display: none;
                backdrop-filter: blur(10px);
            }}
            .objects-list {{
                background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
                border-radius: 10px;
                padding: 15px;
                margin-top: 15px;
                border-left: 4px solid #2196F3;
                max-height: 200px;
                overflow-y: auto;
            }}
            .object-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                margin: 4px 0;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #ddd;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            .object-item:hover {{
                background: #f0f8ff;
                transform: translateX(5px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .object-item.selected {{
                border-left-color: #4CAF50;
                background: #e8f5e8;
            }}
            .wall-object {{
                border-left-color: #8B4513 !important;
            }}
            .floor-object {{
                border-left-color: #DEB887 !important;
            }}
            .custom-object {{
                border-left-color: #FF1493 !important;
            }}
            
            /* Modal Styles */
            .product-modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.7);
                backdrop-filter: blur(5px);
            }}
            .modal-content {{
                background-color: white;
                margin: 2% auto;
                padding: 0;
                border-radius: 15px;
                width: 90%;
                max-width: 1000px;
                max-height: 90vh;
                overflow: hidden;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                animation: modalSlideIn 0.3s ease;
            }}
            @keyframes modalSlideIn {{
                from {{ transform: translateY(-50px); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}
            .modal-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .modal-title {{
                font-size: 24px;
                font-weight: 600;
                margin: 0;
            }}
            .close-btn {{
                background: none;
                border: none;
                color: white;
                font-size: 28px;
                cursor: pointer;
                padding: 0;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s ease;
            }}
            .close-btn:hover {{
                background: rgba(255,255,255,0.2);
            }}
            .modal-body {{
                padding: 20px;
                max-height: 70vh;
                overflow-y: auto;
            }}
            
            /* Carousel Styles for Modal */
            .carousel-container {{
                position: relative;
                margin: 20px 0;
            }}
            .carousel-track {{
                display: flex;
                transition: transform 0.3s ease;
                gap: 20px;
                overflow: hidden;
            }}
            .carousel-card {{
                min-width: 300px;
                max-width: 300px;
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                border: 1px solid #e1e5e9;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                cursor: pointer;
                position: relative;
            }}
            .carousel-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0,0,0,0.2);
            }}
            .product-number {{
                position: absolute;
                top: -10px;
                left: 15px;
                background: #4CAF50;
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 14px;
            }}
            .retailer-badge {{
                background: #e3f2fd;
                color: #1976d2;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: 500;
                margin-bottom: 10px;
                display: inline-block;
            }}
            .carousel-nav {{
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                background: rgba(255,255,255,0.9);
                border: 2px solid #ddd;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                cursor: pointer;
                font-size: 20px;
                color: #333;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .carousel-nav:hover {{
                background: white;
                border-color: #4CAF50;
                color: #4CAF50;
            }}
            .carousel-prev {{
                left: -25px;
            }}
            .carousel-next {{
                right: -25px;
            }}
            .carousel-indicators {{
                display: flex;
                justify-content: center;
                gap: 8px;
                margin-top: 20px;
            }}
            .indicator {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #ddd;
                cursor: pointer;
                transition: background 0.2s ease;
            }}
            .indicator.active {{
                background: #4CAF50;
            }}
            .label-input {{
                position: absolute;
                background: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 10px;
                display: none;
                z-index: 40;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .label-input input {{
                border: 1px solid #ddd;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
                width: 200px;
            }}
            .confirm-btn, .cancel-btn {{
                margin-left: 5px;
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }}
            .confirm-btn {{
                background: #4CAF50;
                color: white;
            }}
            .cancel-btn {{
                background: #f44336;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="controls">
                <button class="btn btn-selection active" id="selectionModeBtn">🎯 Selection Mode</button>
                <button class="btn btn-custom" id="customModeBtn">✏️ Custom Box Mode</button>
                <button class="btn btn-select-all" id="selectAllBtn">📋 Select All</button>
                <button class="btn btn-clear" id="clearAllBtn">🗑️ Clear All</button>
                <button class="btn btn-reset" id="resetBtn">🔄 Reset</button>
                <div class="checkbox-container">
                    <input type="checkbox" id="includeWalls">
                    <label for="includeWalls">🏠 Include walls/flooring</label>
                </div>
            </div>
            
            <div class="canvas-container">
                <canvas id="imageCanvas" width="{display_width}" height="{display_height}"></canvas>
                <div id="modeIndicator" class="mode-indicator">🎯 Selection Mode - Hover & Click</div>
                <div id="tooltip" class="tooltip"></div>
                <div id="labelInput" class="label-input">
                    <input type="text" id="customLabel" placeholder="Enter object name..." maxlength="25">
                    <button class="confirm-btn" onclick="confirmCustomLabel()">✓</button>
                    <button class="cancel-btn" onclick="cancelCustomLabel()">✗</button>
                </div>
            </div>
            
            <div id="selectionSummary" class="selection-summary"></div>
            
            <div class="objects-list" id="objectsList">
                <div><strong>📋 Detected Objects (Click to view products):</strong></div>
            </div>
        </div>

        <!-- Product Modal -->
        <div id="productModal" class="product-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title" id="modalTitle">Product Options</h2>
                    <button class="close-btn" onclick="closeProductModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <div id="carouselContainer" class="carousel-container">
                        <!-- Carousel will be inserted here -->
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Global variables
            const canvas = document.getElementById('imageCanvas');
            const ctx = canvas.getContext('2d');
            const objects = {json.dumps(js_objects)};
            const objectColors = {json.dumps(object_colors)};
            const productsData = {json.dumps(js_products_data)};
            const imageData = "data:image/png;base64,{img_str}";
            const scale = {scale};
            
            let selectedObjects = [];
            let hoveredObject = null;
            let isCustomMode = false;
            let isDrawing = false;
            let customBoxes = [];
            let startPoint = null;
            let currentPoint = null;
            let currentModalSlide = 0;
            
            // Load image
            const img = new Image();
            img.onload = function() {{
                drawCanvas();
                updateObjectsList();
                setupEventListeners();
            }};
            img.src = imageData;
            
            function setupEventListeners() {{
                // Mode buttons
                document.getElementById('selectionModeBtn').addEventListener('click', setSelectionMode);
                document.getElementById('customModeBtn').addEventListener('click', setCustomMode);
                document.getElementById('selectAllBtn').addEventListener('click', selectAllObjects);
                document.getElementById('clearAllBtn').addEventListener('click', clearAllObjects);
                document.getElementById('resetBtn').addEventListener('click', resetAll);
                document.getElementById('includeWalls').addEventListener('change', updateWallsOption);
                
                // Canvas events
                canvas.addEventListener('mousedown', handleMouseDown);
                canvas.addEventListener('mousemove', handleMouseMove);
                canvas.addEventListener('mouseup', handleMouseUp);
                canvas.addEventListener('mouseleave', handleMouseLeave);
                
                // Keyboard events
                document.addEventListener('keydown', handleKeyDown);
                
                // Modal close events
                window.addEventListener('click', function(event) {{
                    const modal = document.getElementById('productModal');
                    if (event.target === modal) {{
                        closeProductModal();
                    }}
                }});
            }}
            
            function setSelectionMode() {{
                isCustomMode = false;
                document.getElementById('selectionModeBtn').classList.add('active');
                document.getElementById('customModeBtn').classList.remove('active');
                document.getElementById('modeIndicator').textContent = '🎯 Selection Mode - Hover & Click';
                canvas.style.cursor = 'crosshair';
            }}
            
            function setCustomMode() {{
                isCustomMode = true;
                document.getElementById('customModeBtn').classList.add('active');
                document.getElementById('selectionModeBtn').classList.remove('active');
                document.getElementById('modeIndicator').textContent = '✏️ Custom Box Mode - Draw to Create';
                canvas.style.cursor = 'crosshair';
            }}
            
            function selectAllObjects() {{
                selectedObjects = [];
                const includeWalls = document.getElementById('includeWalls').checked;
                
                objects.forEach((obj, index) => {{
                    if (includeWalls || (obj.class !== 'wall' && obj.class !== 'floor')) {{
                        selectedObjects.push(index);
                    }}
                }});
                
                customBoxes.forEach((box, index) => {{
                    selectedObjects.push(objects.length + index);
                }});
                
                drawCanvas();
                updateObjectsList();
                updateSelectionSummary();
            }}
            
            function clearAllObjects() {{
                selectedObjects = [];
                drawCanvas();
                updateObjectsList();
                updateSelectionSummary();
            }}
            
            function resetAll() {{
                selectedObjects = [];
                customBoxes = [];
                hoveredObject = null;
                drawCanvas();
                updateObjectsList();
                updateSelectionSummary();
            }}
            
            function updateWallsOption() {{
                updateObjectsList();
            }}
            
            function handleMouseDown(e) {{
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                if (isCustomMode) {{
                    isDrawing = true;
                    startPoint = {{x: x, y: y}};
                    currentPoint = {{x: x, y: y}};
                }} else {{
                    // Selection mode - check for object click
                    const clickedObjectIndex = getObjectAtPoint(x, y);
                    if (clickedObjectIndex !== null) {{
                        // Check if object has products data
                        const objId = objects[clickedObjectIndex]?.id || clickedObjectIndex;
                        if (productsData[objId] && productsData[objId].length > 0) {{
                            showProductModal(objects[clickedObjectIndex], productsData[objId]);
                        }} else {{
                            // No products available, just toggle selection
                            toggleSelection(clickedObjectIndex);
                        }}
                    }}
                }}
            }}
            
            function handleMouseMove(e) {{
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                if (isCustomMode && isDrawing) {{
                    currentPoint = {{x: x, y: y}};
                    drawCanvas();
                    drawTemporaryBox();
                }} else if (!isCustomMode) {{
                    // Update hover state
                    const objectIndex = getObjectAtPoint(x, y);
                    if (objectIndex !== hoveredObject) {{
                        hoveredObject = objectIndex;
                        drawCanvas();
                        updateTooltip(e, objectIndex);
                        
                        // Debug logging - remove in production
                        if (objectIndex !== null && objects[objectIndex]) {{
                            console.log(`Hovering over: ${{objects[objectIndex].class}} (index: ${{objectIndex}})`);
                        }}
                    }}
                }}
            }}
            
            function handleMouseUp(e) {{
                if (isCustomMode && isDrawing) {{
                    isDrawing = false;
                    if (startPoint && currentPoint) {{
                        const width = Math.abs(currentPoint.x - startPoint.x);
                        const height = Math.abs(currentPoint.y - startPoint.y);
                        
                        if (width > 20 && height > 20) {{
                            showLabelInput(e);
                        }}
                    }}
                }}
            }}
            
            function handleMouseLeave() {{
                hoveredObject = null;
                drawCanvas();
                hideTooltip();
            }}
            
            function handleKeyDown(e) {{
                if (e.key === 'Escape') {{
                    closeProductModal();
                    cancelCustomLabel();
                }}
            }}
            
            function getObjectAtPoint(x, y) {{
                // Debug logging
                console.log(`Checking point (${{x}}, ${{y}}) against ${{objects.length}} objects`);
                
                // First check custom boxes (they're usually on top)
                for (let i = customBoxes.length - 1; i >= 0; i--) {{
                    const box = customBoxes[i];
                    if (x >= box.x && x <= box.x + box.width && 
                        y >= box.y && y <= box.y + box.height) {{
                        return objects.length + i;
                    }}
                }}
                
                // Then check objects - prioritize smaller objects (usually on top)
                let candidates = [];
                for (let i = objects.length - 1; i >= 0; i--) {{
                    const obj = objects[i];
                    
                    // Debug log for SAM objects
                    if (obj.source === 'sam' && obj.bbox) {{
                        console.log(`SAM Object ${{i}}: ${{obj['class']}}, bbox: [${{obj.bbox.join(', ')}}], has contours: ${{obj.contours && obj.contours.length > 0}}`);
                    }}
                    
                    // First check bounding box for quick test
                    if (obj.bbox) {{
                        const [xMin, yMin, xMax, yMax] = obj.bbox;
                        if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {{
                            console.log(`Point inside bbox of object ${{i}} (${{obj['class']}})`);
                            // Inside bounding box, now check contours
                            if (obj.contours && obj.contours.length > 0) {{
                                for (const contour of obj.contours) {{
                                    if (pointInPolygon(x, y, contour) || isNearPolygon(x, y, contour, 10)) {{
                                        console.log(`Point matches contour of object ${{i}} (${{obj['class']}})`);
                                        candidates.push({{index: i, area: obj.area || calculatePolygonArea(contour)}});
                                        break;
                                    }}
                                }}
                            }} else {{
                                // No contours but inside bbox
                                console.log(`No contours for object ${{i}} (${{obj['class']}}), using bbox`);
                                candidates.push({{index: i, area: obj.area || ((xMax - xMin) * (yMax - yMin))}});
                            }}
                        }}
                    }} else if (obj.contours && obj.contours.length > 0) {{
                        // No bbox, check contours directly
                        console.log(`Object ${{i}} (${{obj['class']}}) has no bbox, checking contours only`);
                        for (const contour of obj.contours) {{
                            if (pointInPolygon(x, y, contour) || isNearPolygon(x, y, contour, 10)) {{
                                candidates.push({{index: i, area: obj.area || calculatePolygonArea(contour)}});
                                break;
                            }}
                        }}
                    }}
                }}
                
                // Return the smallest object that contains the point
                if (candidates.length > 0) {{
                    console.log(`Found ${{candidates.length}} candidates, selecting smallest`);
                    candidates.sort((a, b) => a.area - b.area);
                    return candidates[0].index;
                }}
                
                console.log('No object found at point');
                return null;
            }}
            
            function pointInPolygon(x, y, polygon) {{
                let inside = false;
                for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {{
                    if (((polygon[i][1] > y) !== (polygon[j][1] > y)) &&
                        (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0])) {{
                        inside = !inside;
                    }}
                }}
                return inside;
            }}
            
            function isNearPolygon(x, y, polygon, threshold) {{
                // Check if point is near any edge of the polygon
                for (let i = 0; i < polygon.length; i++) {{
                    const j = (i + 1) % polygon.length;
                    const dist = distanceToLineSegment(x, y, polygon[i][0], polygon[i][1], polygon[j][0], polygon[j][1]);
                    if (dist <= threshold) {{
                        return true;
                    }}
                }}
                return false;
            }}
            
            function distanceToLineSegment(px, py, x1, y1, x2, y2) {{
                const A = px - x1;
                const B = py - y1;
                const C = x2 - x1;
                const D = y2 - y1;
                
                const dot = A * C + B * D;
                const lenSq = C * C + D * D;
                let param = -1;
                
                if (lenSq !== 0) {{
                    param = dot / lenSq;
                }}
                
                let xx, yy;
                
                if (param < 0) {{
                    xx = x1;
                    yy = y1;
                }} else if (param > 1) {{
                    xx = x2;
                    yy = y2;
                }} else {{
                    xx = x1 + param * C;
                    yy = y1 + param * D;
                }}
                
                const dx = px - xx;
                const dy = py - yy;
                return Math.sqrt(dx * dx + dy * dy);
            }}
            
            function calculatePolygonArea(polygon) {{
                let area = 0;
                for (let i = 0; i < polygon.length; i++) {{
                    const j = (i + 1) % polygon.length;
                    area += polygon[i][0] * polygon[j][1];
                    area -= polygon[j][0] * polygon[i][1];
                }}
                return Math.abs(area / 2);
            }}
            
            function toggleSelection(objIndex) {{
                const index = selectedObjects.indexOf(objIndex);
                if (index > -1) {{
                    selectedObjects.splice(index, 1);
                }} else {{
                    selectedObjects.push(objIndex);
                }}
                drawCanvas();
                updateObjectsList();
                updateSelectionSummary();
            }}
            
            function drawCanvas() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Draw objects
                objects.forEach((obj, index) => {{
                    const isSelected = selectedObjects.includes(index);
                    const isHovered = hoveredObject === index;
                    const color = objectColors[obj.class] || '#FF1493';
                    
                    if (obj.contours && obj.contours.length > 0) {{
                        obj.contours.forEach(contour => {{
                            ctx.beginPath();
                            ctx.moveTo(contour[0][0], contour[0][1]);
                            for (let i = 1; i < contour.length; i++) {{
                                ctx.lineTo(contour[i][0], contour[i][1]);
                            }}
                            ctx.closePath();
                            
                            if (isSelected) {{
                                ctx.fillStyle = color + '40';
                                ctx.fill();
                            }}
                            
                            ctx.strokeStyle = isHovered ? '#FF0000' : (isSelected ? color : 'transparent');
                            ctx.lineWidth = isHovered ? 3 : (isSelected ? 2 : 1);
                            ctx.stroke();
                        }});
                    }}
                }});
                
                // Draw custom boxes
                customBoxes.forEach((box, index) => {{
                    const objIndex = objects.length + index;
                    const isSelected = selectedObjects.includes(objIndex);
                    const isHovered = hoveredObject === objIndex;
                    
                    ctx.strokeStyle = isHovered ? '#FF0000' : (isSelected ? '#FF1493' : '#FF1493');
                    ctx.lineWidth = isHovered ? 3 : 2;
                    ctx.strokeRect(box.x, box.y, box.width, box.height);
                    
                    if (isSelected) {{
                        ctx.fillStyle = '#FF149340';
                        ctx.fillRect(box.x, box.y, box.width, box.height);
                    }}
                }});
            }}
            
            function drawTemporaryBox() {{
                if (startPoint && currentPoint) {{
                    const x = Math.min(startPoint.x, currentPoint.x);
                    const y = Math.min(startPoint.y, currentPoint.y);
                    const width = Math.abs(currentPoint.x - startPoint.x);
                    const height = Math.abs(currentPoint.y - startPoint.y);
                    
                    ctx.strokeStyle = '#FF1493';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(x, y, width, height);
                    ctx.setLineDash([]);
                }}
            }}
            
            function updateTooltip(e, objectIndex) {{
                const tooltip = document.getElementById('tooltip');
                if (objectIndex !== null) {{
                    let obj;
                    if (objectIndex < objects.length) {{
                        obj = objects[objectIndex];
                    }} else {{
                        obj = customBoxes[objectIndex - objects.length];
                    }}
                    
                    const objId = obj?.id || objectIndex;
                    const hasProducts = productsData[objId] && productsData[objId].length > 0;
                    const productText = hasProducts ? ' (Click to view products)' : '';
                    
                    tooltip.textContent = `${{obj.class}} (${{Math.min(100, (obj.confidence * 100)).toFixed(0)}}%)${{productText}}`;
                    tooltip.style.left = e.pageX + 10 + 'px';
                    tooltip.style.top = e.pageY - 30 + 'px';
                    tooltip.style.display = 'block';
                }} else {{
                    hideTooltip();
                }}
            }}
            
            function hideTooltip() {{
                document.getElementById('tooltip').style.display = 'none';
            }}
            
            function updateObjectsList() {{
                const listContainer = document.getElementById('objectsList');
                const includeWalls = document.getElementById('includeWalls').checked;
                listContainer.innerHTML = '<div><strong>📋 Detected Objects (Click to view products):</strong></div>';
                
                objects.forEach((obj, index) => {{
                    if (!includeWalls && (obj.class === 'wall' || obj.class === 'floor')) {{
                        return;
                    }}
                    
                    const isSelected = selectedObjects.includes(index);
                    const objId = obj.id || index;
                    const hasProducts = productsData[objId] && productsData[objId].length > 0;
                    const productCount = hasProducts ? productsData[objId].length : 0;
                    
                    const itemDiv = document.createElement('div');
                    itemDiv.className = `object-item ${{isSelected ? 'selected' : ''}} ${{obj.class === 'wall' ? 'wall-object' : ''}} ${{obj.class === 'floor' ? 'floor-object' : ''}}`;
                    itemDiv.innerHTML = `
                        <span>${{obj.class}} (${{obj.source}}) ${{hasProducts ? `<small style="color: #4CAF50;">[${{productCount}} products]</small>` : ''}}</span>
                        <span>${{Math.min(100, (obj.confidence * 100)).toFixed(0)}}%</span>
                    `;
                    itemDiv.onclick = () => {{
                        if (hasProducts) {{
                            showProductModal(obj, productsData[objId]);
                        }} else {{
                            toggleSelection(index);
                        }}
                    }};
                    listContainer.appendChild(itemDiv);
                }});
                
                // Custom boxes
                customBoxes.forEach((box, index) => {{
                    const objIndex = objects.length + index;
                    const isSelected = selectedObjects.includes(objIndex);
                    const itemDiv = document.createElement('div');
                    itemDiv.className = `object-item custom-object ${{isSelected ? 'selected' : ''}}`;
                    itemDiv.innerHTML = `
                        <span>${{box.class}} (custom)</span>
                        <span>100%</span>
                    `;
                    itemDiv.onclick = () => toggleSelection(objIndex);
                    listContainer.appendChild(itemDiv);
                }});
            }}
            
            function updateSelectionSummary() {{
                const summary = document.getElementById('selectionSummary');
                
                if (selectedObjects.length === 0) {{
                    summary.style.display = 'none';
                    return;
                }}
                
                let summaryText = `Selected Objects (${{selectedObjects.length}})\n`;
                
                selectedObjects.forEach((objIndex, i) => {{
                    let obj;
                    if (objIndex < objects.length) {{
                        obj = objects[objIndex];
                    }} else {{
                        obj = customBoxes[objIndex - objects.length];
                    }}
                    summaryText += `${{i + 1}}. ${{obj.class}} (${{Math.min(100, (obj.confidence * 100)).toFixed(0)}}%)\n`;
                }});
                
                summary.textContent = summaryText;
                summary.style.display = 'block';
            }}
            
            function showProductModal(obj, products) {{
                const modal = document.getElementById('productModal');
                const modalTitle = document.getElementById('modalTitle');
                const carouselContainer = document.getElementById('carouselContainer');
                
                modalTitle.textContent = `🛍️ Products for ${{obj.class}}`;
                
                // Create carousel HTML
                let carouselHTML = `
                    <div class="carousel-track" id="modalCarouselTrack">
                `;
                
                products.forEach((product, index) => {{
                    const title = product.title || 'Unknown Product';
                    const price = product.price || 'Price not available';
                    const retailer = product.retailer || 'Unknown Store';
                    const url = product.url || '#';
                    const score = product.hybrid_score ? ` (Score: ${{product.hybrid_score.toFixed(3)}})` : '';
                    
                    carouselHTML += `
                        <div class="carousel-card" onclick="window.open('${{url}}', '_blank')">
                            <div class="product-number">${{index + 1}}</div>
                            <div class="retailer-badge">${{retailer}}</div>
                            <h4 style="color: #1f77b4; margin: 10px 0 8px 0; font-size: 16px; line-height: 1.3; height: 40px; overflow: hidden;">${{title}}</h4>
                            <div style="color: #2e7d32; font-weight: bold; font-size: 20px; margin: 10px 0;">${{price}}</div>
                            ${{score ? `<div style="font-size: 12px; color: #666; margin-bottom: 10px;">Match${{score}}</div>` : ''}}
                            <div style="background: linear-gradient(45deg, #4CAF50, #45a049); color: white; padding: 8px 16px; border: none; border-radius: 20px; font-size: 14px; font-weight: 500; text-align: center; margin-top: 10px; cursor: pointer; width: 100%;">🔗 View Product</div>
                        </div>
                    `;
                }});
                
                carouselHTML += `
                    </div>
                    <button class="carousel-nav carousel-prev" onclick="moveModalCarousel(-1)">←</button>
                    <button class="carousel-nav carousel-next" onclick="moveModalCarousel(1)">→</button>
                    <div class="carousel-indicators">
                `;
                
                products.forEach((_, index) => {{
                    carouselHTML += `<div class="indicator ${{index === 0 ? 'active' : ''}}" onclick="goToModalSlide(${{index}})"></div>`;
                }});
                
                carouselHTML += `</div>`;
                
                carouselContainer.innerHTML = carouselHTML;
                currentModalSlide = 0;
                modal.style.display = 'block';
            }}
            
            function closeProductModal() {{
                document.getElementById('productModal').style.display = 'none';
            }}
            
            function moveModalCarousel(direction) {{
                const track = document.getElementById('modalCarouselTrack');
                const cards = track.children;
                const totalSlides = cards.length;
                
                currentModalSlide += direction;
                
                if (currentModalSlide < 0) {{
                    currentModalSlide = totalSlides - 1;
                }}
                if (currentModalSlide >= totalSlides) {{
                    currentModalSlide = 0;
                }}
                
                const cardWidth = 320; // 300px + 20px gap
                track.style.transform = `translateX(-${{currentModalSlide * cardWidth}}px)`;
                updateModalIndicators();
            }}
            
            function goToModalSlide(slideIndex) {{
                currentModalSlide = slideIndex;
                const track = document.getElementById('modalCarouselTrack');
                const cardWidth = 320;
                track.style.transform = `translateX(-${{slideIndex * cardWidth}}px)`;
                updateModalIndicators();
            }}
            
            function updateModalIndicators() {{
                const indicators = document.querySelectorAll('.carousel-indicators .indicator');
                indicators.forEach((indicator, index) => {{
                    indicator.classList.toggle('active', index === currentModalSlide);
                }});
            }}
            
            function showLabelInput(e) {{
                const labelInput = document.getElementById('labelInput');
                labelInput.style.left = e.pageX - canvas.getBoundingClientRect().left + 'px';
                labelInput.style.top = e.pageY - canvas.getBoundingClientRect().top + 'px';
                labelInput.style.display = 'block';
                document.getElementById('customLabel').focus();
            }}
            
            function confirmCustomLabel() {{
                const label = document.getElementById('customLabel').value.trim();
                if (label && startPoint && currentPoint) {{
                    const x = Math.min(startPoint.x, currentPoint.x);
                    const y = Math.min(startPoint.y, currentPoint.y);
                    const width = Math.abs(currentPoint.x - startPoint.x);
                    const height = Math.abs(currentPoint.y - startPoint.y);
                    
                    customBoxes.push({{
                        x: x,
                        y: y,
                        width: width,
                        height: height,
                        class: label,
                        confidence: 1.0
                    }});
                    
                    selectedObjects.push(objects.length + customBoxes.length - 1);
                }}
                
                cancelCustomLabel();
                drawCanvas();
                updateObjectsList();
                updateSelectionSummary();
            }}
            
            function cancelCustomLabel() {{
                document.getElementById('labelInput').style.display = 'none';
                document.getElementById('customLabel').value = '';
                startPoint = null;
                currentPoint = null;
                drawCanvas();
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content

def auto_select_furniture_objects():
    """Auto-select furniture items like the original app"""
    furniture_items = ["chair", "sofa", "couch", "dining table", "tv", "bed", "table", "bookshelf", "cabinet", "desk"]
    st.session_state.selected_objects = [obj for obj in st.session_state.detected_objects 
                                        if obj.get("class", "").lower() in furniture_items]
    # Update indices as well
    st.session_state.selected_object_indices = [
        i for i, obj in enumerate(st.session_state.detected_objects)
        if obj.get("class", "").lower() in furniture_items
    ]

def crop_and_save_selected_objects():
    """Crops the selected objects from the original image and saves them."""
    if 'selected_objects' not in st.session_state or not st.session_state.selected_objects:
        st.warning("No objects selected for product search.")
        return

    cropped_object_details = []
    output_dir = os.path.join("output", "crops")
    os.makedirs(output_dir, exist_ok=True)

    original_image_cv = st.session_state.get('image') # This is already a CV2 image (numpy array)
    # Get room style caption once for the entire image
    # Ensure original_image_cv is not None before proceeding
    room_style_data = {"caption": "an unknown style"} # Default if image is None
    if original_image_cv is not None:
        logger.info("Generating room style caption for the main image.")
        room_style_data = get_caption_json(original_image_cv, designer=False)
        logger.info(f"Room style caption generated: {room_style_data.get('caption')}")
    else:
        logger.warning("Original image is None, cannot generate room style caption.")

    for obj_index, obj_data in enumerate(st.session_state.selected_objects):
        bbox = obj_data.get("bbox")
        
        if bbox:
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img_cv = original_image_cv[y_min:y_max, x_min:x_max]
            
            if cropped_img_cv.size == 0:
                logger.warning(f"Skipping empty crop for object {obj_data.get('id', obj_index)} at bbox {bbox}")
                continue

            # Sanitize class name for filename
            class_name = obj_data.get("class", "unknown_object")
            filename_safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
            
            timestamp = int(time.time() * 1000)
            crop_filename = f"{filename_safe_class_name}_{obj_data.get('id', obj_index)}_{timestamp}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            
            try:
                cv2.imwrite(crop_path, cropped_img_cv)
                logger.info(f"✅ Saved crop: {crop_path} for object ID {obj_data.get('id', obj_index)}")
                
                # Update the object in st.session_state.selected_objects directly
                st.session_state.selected_objects[obj_index]["crop_path"] = crop_path

                # Generate and store new designer caption JSON
                final_caption_json = {"style": "N/A", "material": "N/A", "colour": "N/A", "era": "N/A", "caption": "Caption generation failed"}
                try:
                    logger.info(f"Generating designer caption for crop: {crop_path}")
                    # Pass the cropped image (NumPy array) directly to get_caption_json
                    designer_caption_json = get_caption_json(cropped_img_cv, designer=True)
                    logger.info(f"Raw crop JSON for {crop_path}: {designer_caption_json}")
                    
                    # The 'blip_caption' field will now store the most important text part of the JSON for display
                    st.session_state.selected_objects[obj_index]["blip_caption"] = designer_caption_json.get('caption', 'No caption generated.')

                    # CLIP embedding (existing logic)
                    try:
                        embedding = extract_clip_embedding(crop_path)
                        # Convert embedding to list for JSON serializability if needed by Streamlit's state
                        st.session_state.selected_objects[obj_index]["clip_embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                        logger.info(f"✅ Extracted CLIP embedding for {crop_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to extract CLIP embedding for {crop_path}: {e}")
                        st.session_state.selected_objects[obj_index]["clip_embedding"] = None
                    
                    cropped_object_details.append({
                        "id": obj_data.get("id", obj_index),
                        "class": class_name,
                        "confidence": obj_data.get("confidence", 0.8),
                        "crop_path": crop_path,
                        "blip_caption": st.session_state.selected_objects[obj_index].get("blip_caption"),
                        "clip_embedding": st.session_state.selected_objects[obj_index].get("clip_embedding") is not None
                    })

                except Exception as e:
                    logger.error(f"❌ Failed to generate designer caption for {crop_path}: {e}", exc_info=True)
                    st.session_state.selected_objects[obj_index]["designer_caption_json"] = final_caption_json # Store default error json
                    st.session_state.selected_objects[obj_index]["blip_caption"] = "Error: Caption generation failed"

            except Exception as e:
                logger.error(f"❌ Failed to save or process crop {crop_path}: {e}")
        else:
            logger.warning(f"Skipping object {obj_data.get('id', obj_index)} due to missing bounding box.")

    # Display UI elements but don't assign their return values
    if cropped_object_details:
        # These UI elements are for display only - don't assign their results
        st.success(f"✅ Successfully cropped {len(cropped_object_details)} selected objects and generated vision features.")
        st.subheader("🖼️ Cropped Objects and Features")
        cols = st.columns(3)
        for i, detail in enumerate(cropped_object_details):
            with cols[i % 3]:
                st.image(detail["crop_path"], caption=f"{detail['class']} (ID: {detail['id']})", use_container_width=True)
                st.markdown(f"**Caption:** `{detail.get('blip_caption', 'N/A')}`")
    else:
        # Don't assign the result of st.warning
        st.warning("No objects were selected or could be cropped.")

    logger.info(f"crop_and_save_selected_objects explicitly storing {len(cropped_object_details)} items in session_state.")
    # Store results in session state to avoid DeltaGenerator issues
    st.session_state.processed_crop_details = cropped_object_details
    # Return nothing to avoid any DeltaGenerator issues
    return

def display_products(products_by_object, selected_object_details):
    """Displays product search results in cards, organized by selected object."""
    # Implementation of display_products function
    pass



# --- Main Application Logic ---
def main_app():
    """Main function to run the Streamlit application."""
    
    init_session_state()

    st.sidebar.title("Settings")
    
    # Detection Settings
    st.sidebar.subheader("🔍 Detection Settings")
    
    # SAM Detection Controls
    sam_enabled = os.getenv("ENABLE_SAM_DETECTION", "on").lower() in ["on", "true", "1"]
    has_replicate_token = bool(REPLICATE_API_TOKEN)
    
    if has_replicate_token and sam_enabled:
        use_sam = st.sidebar.checkbox(
            "Enable SAM Detection", 
            value=True,
            help="Semantic Segment Anything provides detailed object detection but may take 3-5 minutes on first run (cold start)"
        )
        
        if use_sam:
            sam_timeout = st.sidebar.slider(
                "SAM Timeout (seconds)",
                min_value=180,
                max_value=600,
                value=300,
                step=30,
                help="How long to wait for SAM detection. First run may need 5+ minutes due to cold start."
            )
            
            # Store timeout in session state for use by detector
            st.session_state.sam_timeout = sam_timeout
            
            # Show SAM status info
            st.sidebar.info(
                "ℹ️ **SAM Cold Start Info:**\n"
                "- First run: 3-5 minutes\n"
                "- Subsequent runs: 2-3 minutes\n"
                "- Provides semantic labels like 'Armchair', 'Coffee Table'"
            )
    else:
        st.sidebar.warning("⚠️ SAM Detection disabled\n(check REPLICATE_API_TOKEN)")
    
    # Detection Status Indicator
    if st.sidebar.checkbox("Show Detection Status"):
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            yolo_status = "✅ Ready" if st.session_state.get('object_detector') else "❌ Not initialized"
            st.sidebar.markdown(f"**YOLO:** {yolo_status}")
        
        with col2:
            sam_status = "✅ Enabled" if (has_replicate_token and sam_enabled) else "⚠️ Disabled"
            st.sidebar.markdown(f"**SAM:** {sam_status}")
            
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a room image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of your interior space"
    )
    
    if uploaded_file is not None and not st.session_state.get("processed"):
        # Display spinner while processing
        with st.spinner('Analyzing your image... This may take a moment.'):
            try:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    st.session_state.image_path = tmpfile.name

                image = Image.open(st.session_state.image_path)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                st.session_state.image = image_cv # Store as CV2 image

                # --- Main Detection Logic (Simplified) ---
                # The ObjectDetector now handles YOLO, SAM/Mask2Former, and merging internally.
                detector = get_object_detector()
                segmented_image, detected_objects, segmented_image_path = detector.detect_objects(st.session_state.image_path)
                
                st.session_state.objects = detected_objects
                st.session_state.segmented_image = segmented_image
                st.session_state.segmented_image_path = segmented_image_path
                
                logger.info(f"✅ Detection complete. Found {len(st.session_state.objects)} objects.")

                if st.session_state.objects:
                    st.session_state.processed = True
                
                # Set flag to indicate new results are ready
                st.session_state.new_results_ready = True

            except Exception as e:
                logger.error(f"An error occurred during image processing: {e}", exc_info=True)
                st.error("Sorry, something went wrong during image analysis. Please try again.")
                st.session_state.processed = False
            finally:
                st.session_state.processing = False
        
        # Rerun to update the UI
        if st.session_state.new_results_ready:
            st.rerun()

    # --- UI Rendering ---
    
    # Display the main image and interactive UI
    if st.session_state.get("processed"):
        display_interactive_ui(st)
    elif not st.session_state.get("processing"):
        # Show a placeholder or the original uploaded image if not processing and not processed
        if st.session_state.get('image') is not None:
            st.image(cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB), caption="Your uploaded image", use_container_width=True)

    # Sidebar for object selection and search
    if st.session_state.get("processed"):
        with st.sidebar:
            st.header("🪑 Detected Objects")
            st.write("Click on objects in the list below to select them for a product search.")
            
            # Auto-select furniture objects to start with
            if "selected_objects_indices" not in st.session_state:
                auto_select_furniture_objects()
            
            # Display selectable list of objects
            for i, obj in enumerate(st.session_state.objects):
                is_selected = i in st.session_state.get("selected_objects_indices", [])
                label = f"{obj.get('class', 'Unknown')} (Conf: {obj.get('confidence', 0):.2f})"
                if st.checkbox(label, value=is_selected, key=f"obj_select_{i}"):
                    if i not in st.session_state.selected_objects_indices:
                        st.session_state.selected_objects_indices.append(i)
                elif i in st.session_state.selected_objects_indices:
                    st.session_state.selected_objects_indices.remove(i)
            
            if st.button("🖼️ Find Matching Products", type="primary"):
                with st.spinner("Cropping objects, generating features, and searching for products..."):
                    crop_and_save_selected_objects()
                    st.rerun()

    # Display product search results
    if st.session_state.get("products_by_object"):
        display_products(st.session_state.products_by_object, st.session_state.get("processed_crop_details", []))


def display_interactive_ui(st_instance):
    """Refactored UI display logic for clarity."""
    st_instance.header("Interactive Analysis")
    st_instance.write("Hover over objects to see details, click to select/deselect for product search.")

    # Create and display the interactive component
    if 'interactive_html' not in st_instance.session_state or st_instance.session_state.get('last_image_path') != st_instance.session_state.get('image_path'):
        st_instance.session_state.interactive_html = create_interactive_html(
            st_instance.session_state.image, 
            st_instance.session_state.objects
        )
        st_instance.session_state.last_image_path = st_instance.session_state.get('image_path')

    if st_instance.session_state.interactive_html:
        selected_object_id = components.html(
            st_instance.session_state.interactive_html,
            height=st_instance.session_state.get('image_height', 800) + 20, # Add some padding
            scrolling=False
        )
        
        # Handle object selection from the interactive component
        if selected_object_id and isinstance(selected_object_id, str):
            try:
                # The ID is expected to be in the format 'yolo_i' or 'sam_i'
                obj_index = int(selected_object_id.split('_')[-1])
                
                if 0 <= obj_index < len(st.session_state.objects):
                    if obj_index in st.session_state.selected_objects_indices:
                        st.session_state.selected_objects_indices.remove(obj_index)
                    else:
                        st.session_state.selected_objects_indices.append(obj_index)
                    st.rerun()

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse selected object ID: {selected_object_id}. Error: {e}")

if __name__ == "__main__":
    # Fix asyncio event loop issues on Windows
    try:
        import asyncio
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass
    
    main_app() 