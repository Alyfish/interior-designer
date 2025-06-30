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

# Local imports - Fixed import path
from vision_features import generate_blip_caption, extract_clip_embedding, get_caption_json, get_caption_json_gpt4v, merge_captions, polish_caption_with_openai
from new_object_detector import ObjectDetector, SegBackend
from new_product_matcher import (
    create_new_product_search_agent, 
    parse_agent_response_to_products, 
    search_products_hybrid, 
    search_products_enhanced,
    search_products_with_visual_similarity,
    ENABLE_REVERSE_IMAGE_SEARCH
)
from config import *
from config import REPLICATE_API_TOKEN

# Enhanced recommendation imports (ADDITIVE - does not break existing features)
try:
    from room_context_analyzer import RoomContextAnalyzer
    from session_tracker import SessionTracker
    ENHANCED_RECOMMENDATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced recommendations not available: {e}")
    ENHANCED_RECOMMENDATIONS_AVAILABLE = False

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
def get_object_detector(backend_key: str):
    """Simple singleton for the detector, now backend-aware."""
    try:
        # Map string choice to Enum
        backend_map = {
            "YOLOV8": SegBackend.YOLOV8,
            "MASK2FORMER": SegBackend.MASK2FORMER,
            "COMBINED": SegBackend.COMBINED,
        }
        selected_backend = backend_map.get(backend_key, SegBackend.COMBINED)
        
        detector = ObjectDetector(backend=selected_backend)
        logger.info(f"✅ Object detector initialized successfully with {selected_backend.name} backend")
        return detector
    except Exception as e:
        st.error(f"Failed to initialize Object Detector: {e}")
        logger.error(f"❌ Failed to initialize object detector: {e}")
        return None

@st.cache_resource
def get_product_agent():
    """Get product matching agent"""
    try:
        agent = create_new_product_search_agent()
        logger.info("✅ Product search agent initialized successfully")
        return agent
    except Exception as e:
        st.error(f"Failed to initialize Product Agent: {e}")
        logger.error(f"❌ Failed to initialize product agent: {e}")
        return None

def create_interactive_html(image, objects, products_data=None, debug_mode=False):
    """Create interactive HTML with proper coordinate handling"""
    import json
    import base64
    import io
    from PIL import Image
    
    # Convert image to base64 PNG
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Get actual image dimensions
    img_height, img_width = image.shape[:2]
    
    # Calculate display dimensions with constraints
    max_width = 1200
    max_height = 800
    scale = min(max_width / img_width, max_height / img_height, 1)
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)
    
    logger.info(f"Image scaling - Original: {img_width}x{img_height}, Display: {display_width}x{display_height}, Scale: {scale}")
    
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
    
    # Prepare objects - DON'T scale coordinates here, let JavaScript handle it
    js_objects = []
    for i, obj in enumerate(objects):
        try:
            # Convert numpy types to regular Python types
            def convert_to_python_types(value):
                """Convert numpy types to Python native types for JSON serialization"""
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    return int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    return float(value)
                elif isinstance(value, (list, tuple)):
                    return [convert_to_python_types(v) for v in value]
                elif isinstance(value, dict):
                    return {k: convert_to_python_types(v) for k, v in value.items()}
                else:
                    return value
            
            # Process contours
            contours = obj.get('contours', [])
            if contours:
                contours = convert_to_python_types(contours)
            
            # Process bbox
            bbox = obj.get('bbox', [])
            if bbox:
                bbox = convert_to_python_types(bbox)
            
            obj_data = {
                'id': convert_to_python_types(obj.get('id', i)),
                'class': obj.get('class', 'unknown'),
                'confidence': convert_to_python_types(obj.get('confidence', 0.0)),
                'contours': contours,
                'bbox': bbox,
                'source': obj.get('source', 'detection'),
            }
            js_objects.append(obj_data)
            
        except Exception as e:
            logger.error(f"Error processing object {i}: {e}")
            continue

    # Prepare products data for JavaScript
    js_products_data = {}
    if products_data:
        for obj_id, products in products_data.items():
            js_products_data[str(obj_id)] = products

    # Create the HTML with modal carousel
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Image</title>
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 0; }}
            .container {{ position: relative; max-width: 100%; }}
            
            .controls {{
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
            }}
            
            .btn {{
                background: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s;
            }}
            
            .btn:hover {{
                background: #0056b3;
            }}
            
            .btn.active {{
                background: #28a745;
            }}
            
            .btn-selection {{ background: #007bff; }}
            .btn-custom {{ background: #6f42c1; }}
            .btn-select-all {{ background: #17a2b8; }}
            .btn-clear {{ background: #dc3545; }}
            .btn-reset {{ background: #6c757d; }}
            
            .checkbox-container {{
                display: flex;
                align-items: center;
                gap: 5px;
                margin-left: auto;
            }}
            
            canvas {{ position: absolute; top: 0; left: 0; cursor: crosshair; }}
            
            #tooltip {{
                position: absolute;
                display: none;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 13px;
                pointer-events: none;
                z-index: 1000;
            }}
            
            #custom-box {{
                position: absolute;
                border: 2px dashed #ff4b4b;
                pointer-events: none;
                display: none;
            }}
            
            .mode-indicator {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 100;
            }}
            
            .selection-summary {{
                margin-top: 10px;
                padding: 10px;
                background: #e9ecef;
                border-radius: 4px;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container" id="container">
            <div class="controls">
                <button class="btn btn-selection active" id="selectionModeBtn">🎯 Selection Mode</button>
                <button class="btn btn-custom" id="customModeBtn">✏️ Draw Custom Box</button>
                <button class="btn btn-select-all" id="selectAllBtn">📋 Select All</button>
                <button class="btn btn-clear" id="clearAllBtn">🗑️ Clear All</button>
                <button class="btn btn-reset" id="resetBtn">🔄 Reset</button>
                <div class="checkbox-container">
                    <input type="checkbox" id="includeWalls">
                    <label for="includeWalls">🏠 Include walls/flooring</label>
                </div>
            </div>
            
            <div style="position: relative;">
                <img id="base-image" src="data:image/png;base64,{img_str}" alt="Uploaded image" style="max-width: 100%; display: block;">
                <canvas id="outline-canvas"></canvas>
                <canvas id="highlight-canvas"></canvas>
                <canvas id="hover-canvas"></canvas>
                <div id="tooltip"></div>
                <div id="custom-box"></div>
                <div id="modeIndicator" class="mode-indicator">🎯 Selection Mode</div>
            </div>
            
            <div id="selectionSummary" class="selection-summary"></div>
        </div>

        <script>
            const container = document.getElementById('container');
            const image = document.getElementById('base-image');
            const outlineCanvas = document.getElementById('outline-canvas');
            const highlightCanvas = document.getElementById('highlight-canvas');
            const hoverCanvas = document.getElementById('hover-canvas');
            const tooltip = document.getElementById('tooltip');
            const customBoxDiv = document.getElementById('custom-box');

            // Filter out non-interactive objects like walls, floor, ceiling
            const allObjects = {json.dumps(js_objects)};
            const objects = allObjects;  // Keep all objects for now
            
            if ({str(debug_mode).lower()}) console.log(`Loaded ${{objects.length}} objects`);
            
            let customBoxes = [];
            let isCustomMode = false;
            let isDrawing = false;
            let startPoint = null;
            let currentPoint = null;

            let scaleX, scaleY;
            let selectedObjects = [];  // Array of selected object indices
            let hoveredObjectIndex = null;
            
            // --- Coordinate and Geometry Functions ---
            
            function updateCanvasSize() {{
                const rect = image.getBoundingClientRect();
                [outlineCanvas, highlightCanvas, hoverCanvas].forEach(c => {{
                    c.width = rect.width;
                    c.height = rect.height;
                    c.style.width = rect.width + 'px';
                    c.style.height = rect.height + 'px';
                }});
                
                // Calculate actual scale factors from original to display
                scaleX = rect.width / {img_width};
                scaleY = rect.height / {img_height};
                
                if ({str(debug_mode).lower()}) {{
                    console.log(`Canvas size updated: ${{rect.width}}x${{rect.height}}`);
                    console.log(`Scale factors: ${{scaleX}}x, ${{scaleY}}y`);
                }}
                
                drawAllOutlines();
                drawAllSelections();
            }}

            // --- Mode Functions ---
            
            function setSelectionMode() {{
                isCustomMode = false;
                document.getElementById('selectionModeBtn').classList.add('active');
                document.getElementById('customModeBtn').classList.remove('active');
                document.getElementById('modeIndicator').textContent = '🎯 Selection Mode';
            }}
            
            function setCustomMode() {{
                isCustomMode = true;
                document.getElementById('customModeBtn').classList.add('active');
                document.getElementById('selectionModeBtn').classList.remove('active');
                document.getElementById('modeIndicator').textContent = '✏️ Draw Custom Box Mode';
            }}
            
            function selectAllObjects() {{
                selectedObjects = [];
                const includeWalls = document.getElementById('includeWalls').checked;
                
                objects.forEach((obj, index) => {{
                    const nonInteractive = ['wall', 'floor', 'ceiling', 'door', 'signboard'];
                    if (includeWalls || !nonInteractive.includes(obj.class.toLowerCase())) {{
                        selectedObjects.push(index);
                    }}
                }});
                
                customBoxes.forEach((box, index) => {{
                    selectedObjects.push('custom_' + index);
                }});
                
                drawAllSelections();
                updateSelectionSummary();
                sendSelectionToStreamlit();
            }}
            
            function clearAllObjects() {{
                selectedObjects = [];
                drawAllSelections();
                updateSelectionSummary();
                sendSelectionToStreamlit();
            }}
            
            function resetAll() {{
                selectedObjects = [];
                customBoxes = [];
                hoveredObjectIndex = null;
                const ctx = hoverCanvas.getContext('2d');
                ctx.clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
                drawAllOutlines();
                drawAllSelections();
                updateSelectionSummary();
                sendSelectionToStreamlit();
            }}

            
            // --- Main Application Logic ---
            
            function getObjectAtPoint(x, y) {{
                if ({str(debug_mode).lower()}) console.log(`🎯 Checking point (${{x.toFixed(1)}}, ${{y.toFixed(1)}})`);
                
                let candidates = [];
                
                // Check all objects
                for (let i = 0; i < objects.length; i++) {{
                    const obj = objects[i];
                    let isInside = false;
                    let distance = Infinity;
                    let area = 0;
                    
                    // Check bounding box first (scale from original to display coordinates)
                    if (obj.bbox && obj.bbox.length === 4) {{
                        const [x1, y1, x2, y2] = obj.bbox;
                        const scaledX1 = x1 * scaleX;
                        const scaledY1 = y1 * scaleY;
                        const scaledX2 = x2 * scaleX;
                        const scaledY2 = y2 * scaleY;
                        
                        area = (scaledX2 - scaledX1) * (scaledY2 - scaledY1);
                        
                        if (x >= scaledX1 && x <= scaledX2 && y >= scaledY1 && y <= scaledY2) {{
                            // Inside bounding box - check contours if available
                            if (obj.contours && obj.contours.length > 0) {{
                                for (const contour of obj.contours) {{
                                    // Scale contour points
                                    const scaledContour = contour.map(pt => [pt[0] * scaleX, pt[1] * scaleY]);
                                    if (pointInPolygon(x, y, scaledContour)) {{
                                        isInside = true;
                                        distance = 0;
                                        break;
                                    }}
                                }}
                            }} else {{
                                // No contours, bbox hit is enough
                                isInside = true;
                                distance = 0;
                            }}
                            
                            if (isInside || distance < 10) {{
                                candidates.push({{
                                    index: i,
                                    distance: distance,
                                    area: area,
                                    isInside: isInside,
                                    obj: obj
                                }});
                            }}
                        }}
                    }}
                }}
                
                // Check custom boxes
                for (let i = customBoxes.length - 1; i >= 0; i--) {{
                    const box = customBoxes[i];
                    if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {{
                        candidates.push({{
                            index: 'custom_' + i,
                            distance: 0,
                            area: box.width * box.height,
                            isInside: true,
                            obj: box
                        }});
                    }}
                }}
                
                if (candidates.length === 0) return null;
                
                // Sort by: inside first, then by distance, then by area (smallest first)
                candidates.sort((a, b) => {{
                    if (a.isInside && !b.isInside) return -1;
                    if (!a.isInside && b.isInside) return 1;
                    if (Math.abs(a.distance - b.distance) > 0.1) return a.distance - b.distance;
                    return a.area - b.area;
                }});
                
                return candidates[0].index;
            }}

            // --- Drawing Functions ---

            function drawAllOutlines() {{
                const ctx = outlineCanvas.getContext('2d');
                ctx.clearRect(0, 0, outlineCanvas.width, outlineCanvas.height);
                
                // Only draw debug bounding boxes if debug mode is enabled
                if ({str(debug_mode).lower()}) {{
                    debugDrawBoundingBoxes();
                }}
            }}
            
            function drawAllSelections() {{
                const ctx = highlightCanvas.getContext('2d');
                ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);
                
                // Draw all selected objects
                selectedObjects.forEach(index => {{
                    drawSelection(index);
                }});
            }}
            
            function drawSelection(index) {{
                const ctx = highlightCanvas.getContext('2d');
                
                let obj;
                if (typeof index === 'string' && index.startsWith('custom_')) {{
                    const customIndex = parseInt(index.split('_')[1], 10);
                    const box = customBoxes[customIndex];
                    if (!box) return;
                    
                    // Draw custom box
                    ctx.save();
                    ctx.fillStyle = 'rgba(255, 20, 147, 0.3)';
                    ctx.strokeStyle = '#FF1493';
                    ctx.lineWidth = 2;
                    ctx.fillRect(box.x, box.y, box.width, box.height);
                    ctx.strokeRect(box.x, box.y, box.width, box.height);
                    ctx.restore();
                    return;
                }}
                
                obj = objects[index];
                if (!obj) return;
                
                const color = getColorForIndex(index);
                ctx.save();
                ctx.fillStyle = 'rgba(' + color[0] + ', ' + color[1] + ', ' + color[2] + ', 0.3)';
                ctx.strokeStyle = 'rgb(' + color[0] + ', ' + color[1] + ', ' + color[2] + ')';
                ctx.lineWidth = 2;
                
                // Draw contours if available
                if (obj.contours && obj.contours.length > 0) {{
                    obj.contours.forEach(contour => {{
                        if (contour.length > 2) {{
                            ctx.beginPath();
                            ctx.moveTo(contour[0][0] * scaleX, contour[0][1] * scaleY);
                            for (let i = 1; i < contour.length; i++) {{
                                ctx.lineTo(contour[i][0] * scaleX, contour[i][1] * scaleY);
                            }}
                            ctx.closePath();
                            ctx.fill();
                            ctx.stroke();
                        }}
                    }});
                }} else if (obj.bbox && obj.bbox.length === 4) {{
                    // Use bbox as fallback
                    const [x1, y1, x2, y2] = obj.bbox;
                    const scaledX1 = x1 * scaleX;
                    const scaledY1 = y1 * scaleY;
                    const scaledX2 = x2 * scaleX;
                    const scaledY2 = y2 * scaleY;
                    
                    ctx.fillRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
                    ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
                }}
                
                ctx.restore();
            }}
            
            function drawHover(index) {{
                const ctx = hoverCanvas.getContext('2d');
                ctx.clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
                
                if (index === null) return;
                
                let obj;
                if (typeof index === 'string' && index.startsWith('custom_')) {{
                    const customIndex = parseInt(index.split('_')[1], 10);
                    const box = customBoxes[customIndex];
                    if (!box) return;
                    
                    // Draw hover for custom box
                    ctx.save();
                    ctx.strokeStyle = '#FF0000';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box.x, box.y, box.width, box.height);
                    ctx.restore();
                    return;
                }}
                
                obj = objects[index];
                if (!obj) return;
                
                ctx.save();
                ctx.strokeStyle = '#FF0000';
                ctx.lineWidth = 3;
                
                // Draw contours if available
                if (obj.contours && obj.contours.length > 0) {{
                    obj.contours.forEach(contour => {{
                        if (contour.length > 2) {{
                            ctx.beginPath();
                            ctx.moveTo(contour[0][0] * scaleX, contour[0][1] * scaleY);
                            for (let i = 1; i < contour.length; i++) {{
                                ctx.lineTo(contour[i][0] * scaleX, contour[i][1] * scaleY);
                            }}
                            ctx.closePath();
                            ctx.stroke();
                        }}
                    }});
                }} else if (obj.bbox && obj.bbox.length === 4) {{
                    // Use bbox as fallback
                    const [x1, y1, x2, y2] = obj.bbox;
                    const scaledX1 = x1 * scaleX;
                    const scaledY1 = y1 * scaleY;
                    const scaledX2 = x2 * scaleX;
                    const scaledY2 = y2 * scaleY;
                    
                    ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
                }}
                
                ctx.restore();
            }}
            
            // Add debug function
            function debugDrawBoundingBoxes() {{
                const ctx = outlineCanvas.getContext('2d');
                ctx.save();
                ctx.strokeStyle = 'yellow';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                
                objects.forEach((obj, index) => {{
                    if (obj.bbox && obj.bbox.length === 4) {{
                        const [x1, y1, x2, y2] = obj.bbox;
                        const scaledX1 = x1 * scaleX;
                        const scaledY1 = y1 * scaleY;
                        const scaledX2 = x2 * scaleX;
                        const scaledY2 = y2 * scaleY;
                        
                        ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
                        ctx.fillStyle = 'yellow';
                        ctx.font = '12px Arial';
                        ctx.fillText(`${{obj.class}} (${{obj.source}})`, scaledX1, scaledY1 - 5);
                    }}
                }});
                
                ctx.restore();
            }}

            function updateTooltip(event, objectIndex) {{
                if (objectIndex !== null) {{
                    let objClass;
                    if (typeof objectIndex === 'string' && objectIndex.startsWith('custom_')) {{
                        objClass = 'Custom Selection';
                    }} else {{
                        objClass = objects[objectIndex].class || 'Unknown';
                    }}
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 10) + 'px';
                    tooltip.style.top = (event.clientY + 10) + 'px';
                    tooltip.textContent = objClass;
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }}

            // --- Missing Geometry Functions ---
            
            function pointInPolygon(x, y, polygon) {{
                let inside = false;
                for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {{
                    const xi = polygon[i][0], yi = polygon[i][1];
                    const xj = polygon[j][0], yj = polygon[j][1];
                    
                    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {{
                        inside = !inside;
                    }}
                }}
                return inside;
            }}
            
            function getColorForIndex(index) {{
                const colors = [
                    [255, 107, 107], [78, 205, 196], [69, 183, 209], [150, 206, 180],
                    [255, 238, 173], [255, 154, 0], [237, 117, 57], [95, 39, 205],
                    [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 255, 0],
                    [255, 0, 128], [128, 0, 255], [0, 128, 255], [0, 255, 128]
                ];
                return colors[index % colors.length];
            }}
            
            function updateSelectionSummary() {{
                const summary = document.getElementById('selectionSummary');
                const count = selectedObjects.length;
                if (count > 0) {{
                    const objectTypes = selectedObjects.map(idx => {{
                        if (typeof idx === 'string' && idx.startsWith('custom_')) {{
                            return 'Custom Box';
                        }}
                        return objects[idx]?.class || 'Unknown';
                    }});
                    summary.textContent = `Selected: ${{count}} object(s) - ${{objectTypes.join(', ')}}`;
                }} else {{
                    summary.textContent = 'No objects selected';
                }}
            }}
            
            function sendSelectionToStreamlit() {{
                const selectedData = {{
                    selectedObjects: selectedObjects.map(idx => {{
                        if (typeof idx === 'string' && idx.startsWith('custom_')) {{
                            const customIndex = parseInt(idx.split('_')[1], 10);
                            return customBoxes[customIndex];
                        }}
                        return objects[idx];
                    }}),
                    selectedIndices: selectedObjects,
                    customBoxes: customBoxes
                }};
                
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: selectedData
                }}, '*');
            }}

            // --- Event Handlers ---
            
            document.getElementById('selectionModeBtn').addEventListener('click', setSelectionMode);
            document.getElementById('customModeBtn').addEventListener('click', setCustomMode);
            document.getElementById('selectAllBtn').addEventListener('click', selectAllObjects);
            document.getElementById('clearAllBtn').addEventListener('click', clearAllObjects);
            document.getElementById('resetBtn').addEventListener('click', resetAll);
            
            window.addEventListener('resize', function() {{
                updateCanvasSize();
            }});
            
            let debugMode = {str(debug_mode).lower()};
            
            function handleMouseMove(event) {{
                const rect = hoverCanvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                
                if (isCustomMode && isDrawing) {{
                    currentPoint = {{x: x, y: y}};
                    drawTemporaryBox();
                }} else if (!isCustomMode) {{
                    const hit = getObjectAtPoint(x, y);
                    
                    if (hit !== hoveredObjectIndex) {{
                        hoveredObjectIndex = hit;
                        drawHover(hit);
                        updateTooltip(event, hit);
                    }}
                }}
            }}
            
            function handleMouseDown(event) {{
                const rect = hoverCanvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                
                if (isCustomMode) {{
                    isDrawing = true;
                    startPoint = {{x: x, y: y}};
                    currentPoint = {{x: x, y: y}};
                }}
            }}
            
            function handleMouseUp(event) {{
                if (isCustomMode && isDrawing) {{
                    isDrawing = false;
                    if (startPoint && currentPoint) {{
                        const width = Math.abs(currentPoint.x - startPoint.x);
                        const height = Math.abs(currentPoint.y - startPoint.y);
                        
                        if (width > 20 && height > 20) {{
                            const x = Math.min(startPoint.x, currentPoint.x);
                            const y = Math.min(startPoint.y, currentPoint.y);
                            
                            customBoxes.push({{
                                x: x,
                                y: y,
                                width: width,
                                height: height,
                                class: 'Custom Selection'
                            }});
                            
                            selectedObjects.push('custom_' + (customBoxes.length - 1));
                            drawAllSelections();
                            updateSelectionSummary();
                            sendSelectionToStreamlit();
                        }}
                    }}
                    
                    // Clear temporary box
                    const ctx = hoverCanvas.getContext('2d');
                    ctx.clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
                }}
            }}
            
            function drawTemporaryBox() {{
                const ctx = hoverCanvas.getContext('2d');
                ctx.clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
                
                if (startPoint && currentPoint) {{
                    const x = Math.min(startPoint.x, currentPoint.x);
                    const y = Math.min(startPoint.y, currentPoint.y);
                    const width = Math.abs(currentPoint.x - startPoint.x);
                    const height = Math.abs(currentPoint.y - startPoint.y);
                    
                    ctx.save();
                    ctx.strokeStyle = '#FF1493';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(x, y, width, height);
                    ctx.restore();
                }}
            }}
            
            hoverCanvas.addEventListener('mousemove', handleMouseMove);
            hoverCanvas.addEventListener('mousedown', handleMouseDown);
            hoverCanvas.addEventListener('mouseup', handleMouseUp);
            
            hoverCanvas.addEventListener('mouseleave', function() {{
                hoveredObjectIndex = null;
                const ctx = hoverCanvas.getContext('2d');
                ctx.clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
                updateTooltip(null, null);
            }});
            
            hoverCanvas.addEventListener('click', function(event) {{
                if (isCustomMode) return; // Handled by mouseup
                
                const rect = hoverCanvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                
                const hit = getObjectAtPoint(x, y);
                
                if (hit !== null) {{
                    // Toggle selection
                    const idx = selectedObjects.indexOf(hit);
                    if (idx > -1) {{
                        selectedObjects.splice(idx, 1);
                    }} else {{
                        selectedObjects.push(hit);
                    }}
                    
                    drawAllSelections();
                    updateSelectionSummary();
                    sendSelectionToStreamlit();
                }}
            }});
            
            // Initialize
            function initializeCanvas() {{
                updateCanvasSize();
                drawAllOutlines();
                updateSelectionSummary();
                
                if (debugMode) {{
                    console.log('🎨 Canvas initialized with ' + objects.length + ' objects');
                    console.log('📏 Image dimensions: ' + {img_width} + 'x' + {img_height});
                    console.log('📐 Display dimensions: ' + image.width + 'x' + image.height);
                }}
            }}
            
            if (image.complete) {{
                initializeCanvas();
            }} else {{
                image.onload = initializeCanvas;
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

def crop_and_save_selected_objects(auto_search=False):
    """Crops selected objects from the original image and saves them with vision features."""
    print(f"🔧 crop_and_save_selected_objects called with auto_search={auto_search}")
    print(f"📊 Image available: {st.session_state.image is not None}, Selected objects: {len(st.session_state.selected_objects) if st.session_state.selected_objects else 0}")
    
    if st.session_state.image is None or not st.session_state.selected_objects:
        # Don't assign the result of st.warning to anything - this prevents DeltaGenerator issues
        print("⚠️ No image or no selected objects - aborting crop")
        st.warning("Please upload an image and select objects first.")
        st.session_state.processed_crop_details = []
        return []

    cropped_object_details = []
    output_dir = os.path.join("output", "crops")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    original_image_cv = st.session_state.image # This is already a CV2 image (numpy array)
    # Get room style caption once for the entire image
    # Ensure original_image_cv is not None before proceeding
    room_style_data = {"caption": "an unknown style"} # Default if image is None
    if original_image_cv is not None:
        logger.info("Generating room style caption for the main image.")
        room_style_data = get_caption_json(original_image_cv, designer=False)
        logger.info(f"Room style caption generated: {room_style_data.get('caption')}")
    else:
        logger.warning("Original image is None, cannot generate room style caption.")

    # Add progress tracking
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for obj_index, obj_data in enumerate(st.session_state.selected_objects):
        # Update progress
        progress = (obj_index + 1) / len(st.session_state.selected_objects)
        progress_bar.progress(progress)
        progress_text.text(f"Processing object {obj_index + 1} of {len(st.session_state.selected_objects)}...")
        
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
                    print(f"🎨 Generating caption for object {obj_index}: {class_name}")
                    logger.info(f"Generating designer caption for crop: {crop_path}")
                    # Pass the cropped image (NumPy array) directly to get_caption_json
                    designer_caption_json = get_caption_json(cropped_img_cv, designer=True)
                    print(f"✅ Caption generated: {designer_caption_json.get('caption', 'N/A')[:50]}...")
                    logger.info(f"Raw crop JSON for {crop_path}: {designer_caption_json}")
                    
                    # The 'blip_caption' field will now store the most important text part of the JSON for display
                    st.session_state.selected_objects[obj_index]["blip_caption"] = designer_caption_json.get('caption', 'No caption generated.')
                    st.session_state.selected_objects[obj_index]["caption_data"] = designer_caption_json
                    st.session_state.selected_objects[obj_index]["designer_caption_json"] = designer_caption_json

                    # CLIP embedding (existing logic)
                    try:
                        print(f"🔧 Extracting CLIP embedding for {class_name}...")
                        embedding = extract_clip_embedding(crop_path)
                        # Convert embedding to list for JSON serializability if needed by Streamlit's state
                        st.session_state.selected_objects[obj_index]["clip_embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                        print(f"✅ CLIP embedding extracted successfully")
                        logger.info(f"✅ Extracted CLIP embedding for {crop_path}")
                    except Exception as e:
                        print(f"❌ CLIP embedding failed: {e}")
                        logger.error(f"❌ Failed to extract CLIP embedding for {crop_path}: {e}")
                        st.session_state.selected_objects[obj_index]["clip_embedding"] = None
                    
                    cropped_object_details.append({
                        "id": obj_data.get("id", obj_index),
                        "class": class_name,
                        "confidence": obj_data.get("confidence", 0.8),
                        "crop_path": crop_path,
                        "blip_caption": st.session_state.selected_objects[obj_index].get("blip_caption"),
                        "designer_caption_json": st.session_state.selected_objects[obj_index].get("designer_caption_json"),
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
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()

    # Display UI elements but don't assign their return values
    if cropped_object_details:
        # These UI elements are for display only - don't assign their results
        st.success(f"✅ Successfully cropped {len(cropped_object_details)} selected objects and generated vision features.")
        st.subheader("🖼️ Cropped Objects and Features")
        cols = st.columns(3)
        for i, detail in enumerate(cropped_object_details):
            with cols[i % 3]:
                st.image(detail["crop_path"], caption=f"{detail['class']} (ID: {detail['id']})", use_container_width=True)
                st.markdown(f"**Caption:** `{detail.get('blip_caption', 'N/A')}`") # Changed from BLIP Caption to just Caption
                designer_info = detail.get("designer_caption_json")
                if designer_info and isinstance(designer_info, dict):
                    st.markdown(f"**Style:** `{designer_info.get('style', 'N/A')}`")
                    st.markdown(f"**Material:** `{designer_info.get('material', 'N/A')}`")
                    st.markdown(f"**Colour:** `{designer_info.get('colour', 'N/A')}`")
                
                # Safe CLIP embedding check - handle boolean or None values
                clip_embedding = detail.get('clip_embedding')
                clip_status = "Generated" if clip_embedding else "Failed"
                st.markdown(f"**CLIP Embedding:** `{clip_status}`")
    else:
        # Don't assign the result of st.warning
        st.warning("No objects were selected or could be cropped.")

    logger.info(f"crop_and_save_selected_objects explicitly storing {len(cropped_object_details)} items in session_state.")
    # Store results in session state to avoid DeltaGenerator issues
    st.session_state.processed_crop_details = cropped_object_details
    
    # If auto_search is enabled, automatically trigger product search
    if auto_search and cropped_object_details:
        return cropped_object_details
    
    # Return the processed objects for potential use
    return cropped_object_details

def display_products(products_by_object, selected_object_details):
    """Displays product search results in cards, organized by selected object."""
    # Implementation of display_products function
    pass



# --- Main Application Logic ---
def main_app():
    """Main function to run the Streamlit application."""
    
    init_session_state()
    
    # Initialize enhanced session tracking if available
    if ENHANCED_RECOMMENDATIONS_AVAILABLE:
        SessionTracker.init_session_state()

    st.sidebar.title("Settings")
    
    # Backend selection
    st.sidebar.subheader("⚙️ Detection Backend")
    backend_choice = st.sidebar.selectbox(
        "Choose a detection model:",
        ("COMBINED", "MASK2FORMER", "YOLOV8"),
        index=0, # Default to COMBINED
        help=(
            "**COMBINED**: Best quality. Uses YOLOv8 for speed and Mask2Former for detail. "
            "**MASK2FORMER**: High-quality semantic segmentation (requires API key). "
            "**YOLOV8**: Fast, local detection."
        )
    )
    st.session_state.backend = backend_choice

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
    
    # Add session insights display if enhanced recommendations available
    if ENHANCED_RECOMMENDATIONS_AVAILABLE and st.session_state.get('interaction_history'):
        with st.sidebar.expander("📊 Session Insights", expanded=False):
            insights = SessionTracker.get_session_insights()
            st.metric("Products Viewed", insights['products_viewed'])
            if insights['avg_style_score'] > 0:
                st.metric("Avg Style Match", f"{insights['avg_style_score']:.2f}")
            if insights['avg_context_score'] > 0:
                st.metric("Avg Context Score", f"{insights['avg_context_score']:.2f}")
            st.metric("Session Time", f"{insights['session_duration']}s")
            st.metric("Rooms Analyzed", insights['rooms_analyzed'])
            
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a room image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of your interior space"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Process image button with protection against multiple clicks and cancel option
        if 'detection_in_progress' not in st.session_state:
            st.session_state.detection_in_progress = False
        
        # Create button layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Disable button if detection is in progress
            button_disabled = st.session_state.detection_in_progress
            button_text = "⏳ Processing..." if button_disabled else "🔍 Detect Objects"
            
            detect_clicked = st.button(button_text, key="detect_btn", disabled=button_disabled, use_container_width=True)
        
        with col2:
            # Show cancel button only if detection is in progress
            if st.session_state.detection_in_progress:
                if st.button("❌ Cancel", key="cancel_btn", use_container_width=True, type="secondary"):
                    try:
                        # Try to cancel SAM detection if running
                        detector_instance = get_object_detector(st.session_state.get("backend", "COMBINED"))
                        if detector_instance and hasattr(detector_instance, 'sam_detector') and detector_instance.sam_detector:
                            detector_instance.sam_detector._cancel_global_prediction()
                        st.session_state.detection_in_progress = False
                        st.success("Detection cancelled!")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Cancel attempt: {e}")
                        st.session_state.detection_in_progress = False
                        st.rerun()
            else:
                st.empty()  # Maintain layout
        
        if detect_clicked:
            st.session_state.detection_in_progress = True
            print("\n" + "="*60)
            print("🚀 STARTING OBJECT DETECTION")
            print(f"📁 Backend: {st.session_state.get('backend', 'COMBINED')}")
            print(f"📸 Image: {uploaded_file.name}")
            print("="*60)
            
            with st.spinner("🤖 Processing image for object detection..."):
                try:
                    # Save uploaded file temporarily
                    temp_img_path = "temp_uploaded_image.jpg"
                    print(f"💾 Saving uploaded image to: {temp_img_path}")
                    with open(temp_img_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    print("✅ Image saved successfully")
                    
                    print("🔧 Initializing object detector...")
                    detector_instance = get_object_detector(st.session_state.get("backend", "COMBINED"))
                    if detector_instance:
                        print("✅ Detector initialized, starting detection...")
                        original_image_cv, detected_objects, segmented_image_path = detector_instance.detect_objects(temp_img_path)
                        
                        if detected_objects:
                            print(f"🎯 DETECTION COMPLETE! Found {len(detected_objects)} objects")
                            print("📝 Objects detected:")
                            for i, obj in enumerate(detected_objects):
                                print(f"   {i+1}. {obj.get('class', 'unknown')} (confidence: {obj.get('confidence', 0):.2f}, source: {obj.get('source', 'unknown')})")
                            
                            # Store results
                            st.session_state.image = original_image_cv
                            st.session_state.objects = detected_objects
                            st.session_state.segmented_image_path = segmented_image_path
                            st.session_state.temp_img_path = temp_img_path
                            st.session_state.processed = True
                            
                            # Force clear any cached components to ensure refresh
                            st.session_state.interactive_html = None
                            
                            print("✅ Results stored in session state")
                            print("🔄 Refreshing UI...")
                            logger.info(f"✅ Detection complete. Stored image with shape {original_image_cv.shape} in session.")
                            st.rerun() # Force a rerun to update the UI immediately
                            
                        else:
                            print("❌ No objects detected in image")
                            st.error("❌ Failed to process image - no objects detected")
                            
                except Exception as e:
                    print(f"❌ ERROR during detection: {str(e)}")
                    logger.error(f"❌ Error during object detection: {e}", exc_info=True)
                    st.error(f"❌ Error processing image: {str(e)}")
                finally:
                    print("🏁 Detection process finished")
                    print("="*60 + "\n")
                    st.session_state.detection_in_progress = False
        
        # Display interactive selection UI if processed
        if st.session_state.get("processed") and st.session_state.get("objects"):
            display_interactive_ui(st)

def display_interactive_ui(st_instance):
    """Refactored UI display logic for clarity."""
    
    with st_instance.columns(2)[1]: # Display in the second column
        st_instance.subheader("🎯 Detected Objects")
        if st.session_state.segmented_image_path and os.path.exists(st.session_state.segmented_image_path):
            seg_image = Image.open(st.session_state.segmented_image_path)
            st_instance.image(seg_image, use_container_width=True)
        else:
            st_instance.warning("⚠️ Segmented image not available.")
    
    st_instance.subheader("🛋️ Interactive Object Selection")
    st_instance.markdown("**Hover** over objects to see details, **click** to select/deselect items.")
    
    # Add debug mode control
    debug_mode = st_instance.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show object bounding boxes and mouse coordinates")
    
    # Create and display the interactive component
    if 'interactive_html' not in st.session_state or st.session_state.interactive_html is None:
        st.session_state.interactive_html = create_interactive_html(
            st.session_state.image, st.session_state.objects, debug_mode=debug_mode
        )
    
    component_height = min(st.session_state.image.shape[0] * 1.2, 800)
    selected_data = components.html(
        st.session_state.interactive_html,
        height=component_height,
        scrolling=True
    )
    
    # Handle selection data
    print(f"🔍 Component returned data: {type(selected_data)}")
    print(f"📊 Raw selected_data: {selected_data}")
    
    # For debugging, let's auto-select some furniture items if no selection is made
    if not st.session_state.selected_objects:
        print("🔧 Auto-selecting furniture items...")
        # Limited list for testing - only specific furniture items
        furniture_items = ["chair", "table", "sofa", "cushion", "vase"]
        
        # Items to exclude from auto-selection
        exclude_items = ["wall", "floor", "ceiling", "door", "signboard", "box"]
        
        auto_selected = []
        for obj in st.session_state.objects:
            obj_class = obj.get("class", "").lower()
            # Map couch to sofa for consistency
            if obj_class == "couch":
                obj_class = "sofa"
            # Only include items in our limited test list
            if obj_class in furniture_items:
                auto_selected.append(obj)
                # Limit to 6 items for testing
                if len(auto_selected) >= 6:
                    break
        
        if auto_selected:
            st.session_state.selected_objects = auto_selected
            print(f"✅ Auto-selected {len(auto_selected)} items")
            st.info(f"🧪 Test Mode: Auto-selected {len(auto_selected)} items (chair, table, sofa, cushion, vase only)")
    
    if selected_data and isinstance(selected_data, dict):
        selected_objects = selected_data.get('selectedObjects', [])
        if selected_objects:
            st.session_state.selected_objects = selected_objects
            st.session_state.include_walls = selected_data.get('includeWalls', False)
            print(f"✅ Updated selected objects: {len(selected_objects)} items")
    else:
        print("⚠️ No selection data received from component")

    # Display selected objects and product search integration
    print(f"📊 UI State - Selected objects: {len(st.session_state.selected_objects) if st.session_state.selected_objects else 0}")
    
    if st.session_state.selected_objects:
        st.info(f"✅ {len(st.session_state.selected_objects)} objects selected")
        print("✅ Showing Find Products button...")
        
        # Single button to process and search products
        if st.button("🔍 Find Matching Products", key="find_products_btn", type="primary", use_container_width=True):
            print("\n" + "="*60)
            print("🛒 STARTING PRODUCT SEARCH")
            print(f"📦 Selected objects: {len(st.session_state.selected_objects)}")
            print("="*60)
            
            with st.spinner("Processing objects and searching for products..."):
                # Step 1: Process objects (crop and extract features)
                print("🔧 Step 1: Processing objects (cropping & feature extraction)...")
                print(f"📊 Selected objects breakdown:")
                object_classes = {}
                for obj in st.session_state.selected_objects:
                    cls = obj.get('class', 'unknown')
                    object_classes[cls] = object_classes.get(cls, 0) + 1
                for cls, count in sorted(object_classes.items()):
                    print(f"   - {cls}: {count}")
                
                processed_objects = crop_and_save_selected_objects(auto_search=True)
                
                if processed_objects:
                    print(f"✅ Processed {len(processed_objects)} objects successfully")
                    
                    # ADD: Extract room context for better recommendations (if available)
                    room_context = None
                    if ENHANCED_RECOMMENDATIONS_AVAILABLE and st.session_state.image is not None:
                        print("🏠 Analyzing room context for enhanced recommendations...")
                        room_analyzer = RoomContextAnalyzer()
                        room_context = room_analyzer.analyze_room_context(
                            st.session_state.image,
                            st.session_state.selected_objects
                        )
                        
                        # Track room upload in session
                        SessionTracker.track_room_upload(room_context)
                        
                        # Store in session state for later use
                        st.session_state.room_context = room_context
                        
                        # Display room insights (optional)
                        with st.expander("🏠 Room Analysis", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Room Type", room_context['room_type'].title())
                                st.metric("Brightness", room_context['room_brightness'].title())
                            with col2:
                                st.metric("Furniture Count", room_context['object_density'])
                                st.metric("Layout", room_context['spatial_layout']['distribution'].title())
                            with col3:
                                st.markdown("**Dominant Colors:**")
                                colors_html = ""
                                for color in room_context['dominant_colors'][:3]:
                                    colors_html += f'<span style="background-color: rgb{color}; padding: 2px 10px; margin: 2px; display: inline-block;">⬤</span>'
                                st.markdown(colors_html, unsafe_allow_html=True)
                    
                    # Step 2: Search for products using the full power of new_product_matcher
                    print("🔧 Step 2: Initializing product search...")
                    from new_product_matcher import search_products_with_visual_similarity, ENABLE_REVERSE_IMAGE_SEARCH
                    
                    # Determine search method based on reverse image search availability
                    if ENABLE_REVERSE_IMAGE_SEARCH and any(obj.get('crop_path') for obj in st.session_state.selected_objects):
                        search_method = 'hybrid'
                        print("🔍 Using HYBRID search (visual + text)")
                        st.info("🔍 Using hybrid search (visual + text) for better results!")
                    else:
                        search_method = 'text_only'
                        print("📝 Using TEXT-ONLY search")
                        if not ENABLE_REVERSE_IMAGE_SEARCH:
                            st.warning("💡 Enable reverse image search in .env for better results!")
                    
                    # Use enhanced search pipeline with new_product_matcher integration
                    print("🔧 Step 3: Running enhanced search pipeline...")
                    from utils.enhanced_product_search import create_enhanced_search_pipeline
                    search_results = create_enhanced_search_pipeline(
                        st.session_state.selected_objects,
                        search_method=search_method,
                        use_caching=True
                    )
                    
                    print(f"🎯 PRODUCT SEARCH COMPLETE! Found results for {len(search_results)} objects")
                    print("📊 Search results summary:")
                    # search_results is a list, not a dict
                    for idx, result in enumerate(search_results):
                        if isinstance(result, dict) and 'products' in result:
                            obj_id = result.get('object_id', idx)
                            products = result.get('products', [])
                            print(f"   Object {obj_id}: {len(products)} products found")
                        else:
                            print(f"   Object {idx}: Unknown result format")
                    
                    # Enhance results with context scoring if available
                    if ENHANCED_RECOMMENDATIONS_AVAILABLE and room_context:
                        print("🎨 Enhancing results with style compatibility scoring...")
                        from utils.enhanced_product_search import enhance_search_results_with_context
                        search_results = enhance_search_results_with_context(
                            search_results,
                            room_context,
                            st.session_state.selected_objects
                        )
                        print("✅ Style scoring applied to search results")
                    
                    # Store results
                    st.session_state.product_search_results = search_results
                    print("✅ Results stored in session state")
                    print("🔄 Refreshing UI...")
                    print("🏁 Product search process finished")
                    print("="*60 + "\n")
                    st.success(f"✅ Found products for {len(search_results)} objects!")
                    st.rerun()  # Refresh to show results
                else:
                    print("❌ No objects were processed successfully")
                    print("🏁 Product search process finished (failed)")
                    print("="*60 + "\n")
        
        # Display results if available
        if st.session_state.get('product_search_results'):
            st.markdown("---")
            st.subheader("🛒 Product Matches")
            
            # Import display module
            from utils.object_product_integration import display_all_objects_with_products
            
            # Add search settings in an expander
            with st.expander("⚙️ Search Settings", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    new_search_method = st.selectbox(
                        "Search Method",
                        ["hybrid", "text_only", "visual_only"],
                        help="Hybrid combines text and visual search"
                    )
                with col2:
                    if st.button("🔄 Re-search", key="research_btn"):
                        with st.spinner("Re-searching with new settings..."):
                            from utils.enhanced_product_search import create_enhanced_search_pipeline
                            search_results = create_enhanced_search_pipeline(
                                st.session_state.selected_objects,
                                search_method=new_search_method,
                                use_caching=False  # Don't use cache for re-search
                            )
                            st.session_state.product_search_results = search_results
                            st.rerun()
            
            # Display all objects with their products (with enhanced tracking if available)
            if ENHANCED_RECOMMENDATIONS_AVAILABLE:
                try:
                    from enhanced_display_integration import display_all_objects_with_products_enhanced
                    display_all_objects_with_products_enhanced(
                        st.session_state.selected_objects,
                        st.session_state.product_search_results
                    )
                except:
                    # Fallback to original display
                    display_all_objects_with_products(
                        st.session_state.selected_objects,
                        st.session_state.product_search_results
                    )
            else:
                display_all_objects_with_products(
                    st.session_state.selected_objects,
                    st.session_state.product_search_results
                )
    else:
        st.info("👆 Select objects from the image above to search for matching products")

if __name__ == "__main__":
    # Fix asyncio event loop issues on Windows
    try:
        import asyncio
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
            pass

if __name__ == "__main__":
    main_app() 